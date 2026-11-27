//! The browser page of `web-batch-frames`: the code that only exists on `wasm32`.

use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use gust::batch::{FrameTime, FrameTiming, TimingInfo};
use gust_wgpu::{DeviceOptions, HeadlessDevice};
use js_sys::{Function, Object, Promise, Reflect};
use wasm_bindgen::{JsCast, JsValue, prelude::wasm_bindgen};
use wasm_bindgen_futures::JsFuture;

use crate::{
    LENGTH, TIMED_STAGES, build, counted_reference, gathered_reference, input, mixed_reference,
    reference, rotated_reference, rows_of, wide_input, wide_reference,
};

/// The storage buffers per stage the raised device asks for: what Chrome 158 reports as the adapter limit on the two
/// adapters measured, so an adapter that offers fewer fails the page.
const RAISED_STORAGE_BUFFERS: u32 = 16;

/// The counts of the three frames' counted stage: a part of the rows, fewer than before (the older rows stay), and a
/// count above the capacity (clamped).
const COUNTS: [u32; 3] = [300, 100, 5000];

/// The last part of the page's `GUST_OK` line; `cargo xtask check-web` requires it.
const CLOSING: &str = "two frames in flight exact, wide stage refused on 8 and exact on 16";

/// The middle part of the line when the browser granted timestamp queries and the third frame's time was read and
/// checked; `cargo xtask check-web` accepts only the middle parts listed in `infos` of the page's `Example` in xtask.
const TIME_READ: &str = "frame time read";

/// The middle part when it did not: the batch reported no timing and the frame's time resolved to `None`.
const TIME_ABSENT: &str = "frame time absent";

/// How far the GPU span of the timed frame may exceed the CPU wall time measured around it: the resolution of
/// `Date.now()`, one millisecond. The wall time is only a coarse bound: the step-zero probe (which read `performance.now()`
/// around single frames) saw about 5 ms in every case on Chrome 158, Radeon 890M only; RTX 5090 pending, a browser timer
/// floor. This page's wall time covers a longer stretch, so the check catches a wild value, not a good-looking wrong one.
const WALL_SLACK_MS: f64 = 1.0;

/// Runs the page and returns the line it logged: `GUST_OK <vendor> <architecture> | <TIME_READ or TIME_ABSENT> |
/// <CLOSING>` or `GUST_FAIL <reason>`. It never rejects; a panic is logged as `GUST_FAIL panic: ...` first.
#[wasm_bindgen]
pub async fn run() -> String {
    std::panic::set_hook(Box::new(|info| {
        log(&format!("GUST_FAIL panic: {}", one_line(&info.to_string())));
    }));
    let line = match frames_in_flight().await {
        Ok(time) => format!(
            "GUST_OK {} | {time} | {CLOSING}",
            browser_adapter_label().await
        ),
        Err(reason) => format!("GUST_FAIL {}", one_line(&reason)),
    };
    log(&line);
    line
}

fn log(line: &str) {
    web_sys::console::log_1(&JsValue::from_str(line));
}

/// The runner reads one line per result, so a reason never spans lines.
fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The vendor and architecture words of the browser's adapter, or `unknown` for a word it does not give. wgpu does not
/// expose the browser adapter it chose, so this asks for one the way `HeadlessDevice::new_async` does.
async fn browser_adapter_label() -> String {
    async fn words() -> Result<String, JsValue> {
        let navigator = Reflect::get(&js_sys::global(), &JsValue::from_str("navigator"))?;
        let gpu = Reflect::get(&navigator, &JsValue::from_str("gpu"))?;
        let request: Function =
            Reflect::get(&gpu, &JsValue::from_str("requestAdapter"))?.dyn_into()?;
        let options = Object::new();
        Reflect::set(
            &options,
            &JsValue::from_str("powerPreference"),
            &JsValue::from_str("high-performance"),
        )?;
        let adapter = JsFuture::from(request.call1(&gpu, &options)?.dyn_into::<Promise>()?).await?;
        let info = Reflect::get(&adapter, &JsValue::from_str("info"))?;
        let word = |name: &str| {
            Reflect::get(&info, &JsValue::from_str(name))
                .ok()
                .and_then(|value| value.as_string())
                .map(|text| one_line(&text).replace(' ', "-"))
                .filter(|text| !text.is_empty())
                .unwrap_or_else(|| "unknown".to_owned())
        };
        Ok(format!("{} {}", word("vendor"), word("architecture")))
    }
    words()
        .await
        .unwrap_or_else(|_| "unknown unknown".to_owned())
}

/// Polls `future` once, with a waker that does nothing: `Some` when it was ready, `None` when it was pending. In the
/// browser nothing a future waits for can happen before the page returns to the event loop, so a future that is pending
/// here is pending for a reason, not by a race.
fn ready_now<F: Future>(future: F) -> Option<F::Output> {
    let mut future = pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    match future.as_mut().poll(&mut context) {
        Poll::Ready(value) => Some(value),
        Poll::Pending => None,
    }
}

fn first_difference(got: &[u32], expected: &[u32]) -> Option<usize> {
    got.iter()
        .zip(expected)
        .position(|(got, want)| got != want)
        .or_else(|| (got.len() != expected.len()).then_some(got.len().min(expected.len())))
}

fn require_exact(what: &str, got: &[u32], expected: &[u32]) -> Result<(), String> {
    match first_difference(got, expected) {
        None => Ok(()),
        Some(first) => Err(format!(
            "{what}: {} values read back, {} expected, first difference at index {first}",
            got.len(),
            expected.len()
        )),
    }
}

/// The timed frame's time, awaited: with the grant (`granted`) it is a value that has [`TIMED_STAGES`] entries, one per
/// stage with a pass, each naming a distinct stage and none longer than the span, and a span within the CPU wall time
/// since `started_ms` (see [`WALL_SLACK_MS`]), and the batch reports the level it was built with and a 1 ns tick;
/// without it the time is `None` and the batch reports no timing. Returns the middle part of the label for the branch
/// that ran.
async fn check_frame_time(
    granted: bool,
    info: Option<TimingInfo>,
    time: FrameTime,
    started_ms: f64,
) -> Result<&'static str, String> {
    let times = time
        .await
        .map_err(|error| format!("frame time failed: {error}"))?;
    let wall_ms = js_sys::Date::now() - started_ms;
    if !granted {
        return match (times, info) {
            (None, None) => Ok(TIME_ABSENT),
            (times, info) => Err(format!(
                "timestamp queries were not granted, but the frame time is {times:?} and the batch timing {info:?}"
            )),
        };
    }
    let Some(info) = info else {
        return Err("timestamp queries were granted, but the batch reports no timing".to_owned());
    };
    if (info.level, info.tick) != (FrameTiming::Passes, Duration::from_nanos(1)) {
        return Err(format!(
            "the batch reports {:?} at a tick of {:?}, not Passes at 1 ns",
            info.level, info.tick
        ));
    }
    let Some(times) = times else {
        return Err(
            "timestamp queries were granted, but the frame time resolved to None".to_owned(),
        );
    };
    // One entry for each stage that has a pass, each naming its own stage, and none longer than the whole span (the
    // browser's rounding keeps the order of the values, so this holds exactly). A span of 0 is not refused: a short
    // frame can read as 0 steps.
    let mut names: Vec<&str> = times
        .passes
        .iter()
        .flat_map(|pass| pass.stages.iter().map(String::as_str))
        .collect();
    let named = names.len();
    names.sort_unstable();
    names.dedup();
    if times.passes.len() != TIMED_STAGES || named != TIMED_STAGES || names.len() != TIMED_STAGES {
        return Err(format!(
            "the frame time has {} entries naming {named} stages ({} distinct), not {TIMED_STAGES} of each: {:?}",
            times.passes.len(),
            names.len(),
            times.passes
        ));
    }
    if let Some(pass) = times.passes.iter().find(|pass| pass.duration > times.span) {
        return Err(format!(
            "the entry of {:?} lasts {:?}, longer than the whole span of {:?}",
            pass.stages, pass.duration, times.span
        ));
    }
    let span_ms = times.span.as_secs_f64() * 1000.0;
    if span_ms > wall_ms + WALL_SLACK_MS {
        return Err(format!(
            "the GPU span of the frame, {span_ms} ms, is longer than the {wall_ms} ms of CPU wall time around it"
        ));
    }
    Ok(TIME_READ)
}

async fn frames_in_flight() -> Result<&'static str, String> {
    // The batch takes its device as an `Arc`; a browser page is one thread, so the handles never cross one.
    #[allow(clippy::arc_with_non_send_sync)]
    let device = Arc::new(
        HeadlessDevice::new_async(wgpu::Backends::BROWSER_WEBGPU)
            .await
            .map_err(|error| format!("no WebGPU device: {error}"))?,
    );
    let (mut batch, h) =
        build(&device, LENGTH).map_err(|error| format!("build failed: {error}"))?;

    // Frame N: its own input and gain, one ticket on the result.
    let (x0, gain0) = (input(1), 5);
    let mut frame0 = ready_now(batch.begin())
        .ok_or("begin of the first frame must return at once")?
        .map_err(|error| format!("begin of frame N failed: {error}"))?;
    frame0
        .write(&h.x, &x0)
        .map_err(|error| format!("frame N write: {error}"))?;
    frame0
        .set(&h.gain, gain0)
        .map_err(|error| format!("frame N set: {error}"))?;
    frame0
        .write(&h.count, &[COUNTS[0]])
        .map_err(|error| format!("frame N count: {error}"))?;
    let mut ticket0 = frame0
        .ticket(&h.y)
        .map_err(|error| format!("frame N ticket: {error}"))?;
    let counted0 = frame0
        .ticket(&h.z)
        .map_err(|error| format!("frame N counted ticket: {error}"))?;
    let _submitted0 = frame0
        .submit()
        .map_err(|error| format!("frame N submit: {error}"))?;

    // Frame N+1 is recorded and submitted while frame N is in flight: a different input and gain, in the same buffers.
    // One frame is in flight and the depth is two, so `begin` returns at once.
    let (x1, gain1) = (input(2), 9);
    let mut frame1 = ready_now(batch.begin())
        .ok_or(
            "begin of frame N+1 must return at once: one frame is in flight and the depth is two",
        )?
        .map_err(|error| format!("begin of frame N+1 failed: {error}"))?;
    frame1
        .write(&h.x, &x1)
        .map_err(|error| format!("frame N+1 write: {error}"))?;
    frame1
        .set(&h.gain, gain1)
        .map_err(|error| format!("frame N+1 set: {error}"))?;
    frame1
        .write(&h.count, &[COUNTS[1]])
        .map_err(|error| format!("frame N+1 count: {error}"))?;
    let ticket1 = frame1
        .ticket(&h.y)
        .map_err(|error| format!("frame N+1 ticket: {error}"))?;
    let counted1 = frame1
        .ticket(&h.z)
        .map_err(|error| format!("frame N+1 counted ticket: {error}"))?;
    let _submitted1 = frame1
        .submit()
        .map_err(|error| format!("frame N+1 submit: {error}"))?;

    // Nothing has returned to the event loop since frame N's submission, so its ticket cannot have resolved.
    if ready_now(&mut ticket0).is_some() {
        return Err(
            "frame N's ticket was resolved when frame N+1 was submitted: it must be pending"
                .to_owned(),
        );
    }
    // Two frames are in flight, so a third `begin` has to wait for the oldest; it is dropped here (the claim goes back).
    if ready_now(batch.begin()).is_some() {
        return Err("a third begin returned at once with two frames in flight".to_owned());
    }

    let values0 = ticket0
        .await
        .map_err(|error| format!("frame N ticket failed: {error}"))?;
    let values1 = ticket1
        .await
        .map_err(|error| format!("frame N+1 ticket failed: {error}"))?;
    require_exact("frame N", &values0, &reference(&x0, gain0))?;
    require_exact("frame N+1", &values1, &reference(&x1, gain1))?;

    // The third frame's `begin` learns from `on_submitted_work_done` that the oldest frame has completed.
    let (x2, gain2) = (input(3), 11);
    let mut frame2 = batch
        .begin()
        .await
        .map_err(|error| format!("begin of frame N+2 failed: {error}"))?;
    frame2
        .write(&h.x, &x2)
        .map_err(|error| format!("frame N+2 write: {error}"))?;
    frame2
        .set(&h.gain, gain2)
        .map_err(|error| format!("frame N+2 set: {error}"))?;
    frame2
        .write(&h.count, &[COUNTS[2]])
        .map_err(|error| format!("frame N+2 count: {error}"))?;
    let rows2 = rows_of(3);
    frame2
        .write(&h.rows, &rows2)
        .map_err(|error| format!("frame N+2 rows: {error}"))?;
    let ticket2 = frame2
        .ticket(&h.y)
        .map_err(|error| format!("frame N+2 ticket: {error}"))?;
    let mixed2 = frame2
        .ticket(&h.mixed)
        .map_err(|error| format!("frame N+2 struct ticket: {error}"))?;
    let rotated2 = frame2
        .ticket(&h.rotated)
        .map_err(|error| format!("frame N+2 struct result ticket: {error}"))?;
    let gathered2 = frame2
        .ticket(&h.gathered)
        .map_err(|error| format!("frame N+2 gathered rows ticket: {error}"))?;
    let counted2 = frame2
        .ticket(&h.z)
        .map_err(|error| format!("frame N+2 counted ticket: {error}"))?;
    // This frame asks for its GPU time (the batch is built with `FrameTiming::Passes`); it is read after the tickets.
    let time2 = frame2.time();
    let timing = batch.frame_timing();
    let granted = device
        .wgpu_device()
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY);
    let started_ms = js_sys::Date::now();
    let _submitted2 = frame2
        .submit()
        .map_err(|error| format!("frame N+2 submit: {error}"))?;
    let values2 = ticket2
        .await
        .map_err(|error| format!("frame N+2 ticket failed: {error}"))?;
    require_exact("frame N+2", &values2, &reference(&x2, gain2))?;

    // The struct stages: a field read and a whole-row result, from a struct buffer the browser's kernels bind as is.
    let mixed = mixed2
        .await
        .map_err(|error| format!("struct field ticket failed: {error}"))?;
    require_exact("struct field read", &mixed, &mixed_reference(&rows2))?;
    let rotated = rotated2
        .await
        .map_err(|error| format!("struct result ticket failed: {error}"))?;
    if rotated != rotated_reference(&rows2) {
        return Err("struct result: the rotated rows differ from the host computation".to_owned());
    }
    // Whole rows through an index handle: a counted stage gathers the rows reversed, through a stage-built index.
    let gathered = gathered2
        .await
        .map_err(|error| format!("gathered rows ticket failed: {error}"))?;
    if gathered != gathered_reference(&rows2, COUNTS[2]) {
        return Err("gathered rows: they differ from the host computation".to_owned());
    }

    // The counted stage (bounded launch in a browser): each frame's rows below its count are that frame's, the rest
    // keep the last frame's values, and a count above the capacity is the capacity. The model is applied in
    // submission order, which is the order the frames ran.
    let mut state = vec![0u32; LENGTH as usize];
    for (what, ticket, x, count) in [
        ("frame N counted", counted0, &x0, COUNTS[0]),
        ("frame N+1 counted", counted1, &x1, COUNTS[1]),
        ("frame N+2 counted", counted2, &x2, COUNTS[2]),
    ] {
        counted_reference(&mut state, x, count);
        let values = ticket
            .await
            .map_err(|error| format!("{what} ticket failed: {error}"))?;
        require_exact(what, &values, &state)?;
    }
    let time = check_frame_time(granted, timing, time2, started_ms).await?;
    wide_stage_is_refused_by_name(&device)?;
    wide_stage_runs_on_a_raised_device().await?;
    Ok(time)
}

/// A stage of ten storage buffers is refused when the batch is built (m14.lim-b, D280): a browser device opened without
/// the option has eight, so the refusal is the typed one, names the stage and the batch, and carries both cures.
fn wide_stage_is_refused_by_name(device: &Arc<HeadlessDevice>) -> Result<(), String> {
    let Err(error) = crate::build_wide(device) else {
        return Err(
            "a stage of ten storage buffers was built on a browser device opened without the option"
                .to_owned(),
        );
    };
    let gust::batch::Error::StageDevice {
        batch,
        stage,
        source,
    } = &error
    else {
        return Err(format!(
            "the wide stage was refused, but not by name: {error}"
        ));
    };
    match source.downcast_ref::<gust_wgpu::Error>() {
        Some(gust_wgpu::Error::DeviceLimit {
            limit: "max_storage_buffers_per_shader_stage",
            requested: 10,
            maximum: 8,
            ..
        }) if (batch.as_str(), stage.as_str()) == ("wide", "out") => Ok(()),
        other => Err(format!(
            "the wide stage was refused with {other:?}, not DeviceLimit 10 against 8 for stage `out` of batch `wide`"
        )),
    }
}

/// The same stage on a device opened with `with_storage_buffers(16)` (m14.lim-web): the page asks for sixteen, the
/// adapter offers them (an adapter that offers fewer fails the open, which is this page's failure), the batch builds, and
/// the result of one frame equals the host computation exactly.
async fn wide_stage_runs_on_a_raised_device() -> Result<(), String> {
    #[allow(clippy::arc_with_non_send_sync)]
    let device = Arc::new(
        HeadlessDevice::new_async_with(
            DeviceOptions::default().with_storage_buffers(RAISED_STORAGE_BUFFERS),
        )
        .await
        .map_err(|error| {
            format!("no WebGPU device with {RAISED_STORAGE_BUFFERS} storage buffers: {error}")
        })?,
    );
    let limit = device.limits().max_storage_buffers_per_shader_stage;
    if limit != RAISED_STORAGE_BUFFERS {
        return Err(format!(
            "the raised device reports {limit} storage buffers per stage, not {RAISED_STORAGE_BUFFERS}"
        ));
    }
    let (mut batch, h) = crate::build_wide(&device)
        .map_err(|error| format!("the wide stage did not build on the raised device: {error}"))?;
    let mut frame = batch
        .begin()
        .await
        .map_err(|error| format!("wide begin failed: {error}"))?;
    for (k, stage) in (0u32..).zip(&h.inputs) {
        frame
            .write(stage, &wide_input(k))
            .map_err(|error| format!("wide write of input {k}: {error}"))?;
    }
    let ticket = frame
        .ticket(&h.out)
        .map_err(|error| format!("wide ticket: {error}"))?;
    let _submitted = frame
        .submit()
        .map_err(|error| format!("wide submit: {error}"))?;
    let values = ticket
        .await
        .map_err(|error| format!("wide ticket failed: {error}"))?;
    require_exact(
        "wide stage on the raised device",
        &values,
        &wide_reference(),
    )
}
