//! The browser page of `web-batch-frames`: the code that only exists on `wasm32`.

use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use gust_wgpu::HeadlessDevice;
use js_sys::{Function, Object, Promise, Reflect};
use wasm_bindgen::{JsCast, JsValue, prelude::wasm_bindgen};
use wasm_bindgen_futures::JsFuture;

use crate::{LENGTH, build, input, reference};

/// The last part of the page's `GUST_OK` line; `cargo xtask check-web` requires it.
const CLOSING: &str = "two frames in flight exact";

/// Runs the page and returns the line it logged: `GUST_OK <vendor> <architecture> | two frames in flight exact` or
/// `GUST_FAIL <reason>`. It never rejects; a panic is logged as `GUST_FAIL panic: ...` first.
#[wasm_bindgen]
pub async fn run() -> String {
    std::panic::set_hook(Box::new(|info| {
        log(&format!("GUST_FAIL panic: {}", one_line(&info.to_string())));
    }));
    let line = match frames_in_flight().await {
        Ok(()) => format!("GUST_OK {} | {CLOSING}", browser_adapter_label().await),
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

async fn frames_in_flight() -> Result<(), String> {
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
    let mut ticket0 = frame0
        .ticket(&h.y)
        .map_err(|error| format!("frame N ticket: {error}"))?;
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
    let ticket1 = frame1
        .ticket(&h.y)
        .map_err(|error| format!("frame N+1 ticket: {error}"))?;
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
    let ticket2 = frame2
        .ticket(&h.y)
        .map_err(|error| format!("frame N+2 ticket: {error}"))?;
    let _submitted2 = frame2
        .submit()
        .map_err(|error| format!("frame N+2 submit: {error}"))?;
    let values2 = ticket2
        .await
        .map_err(|error| format!("frame N+2 ticket failed: {error}"))?;
    require_exact("frame N+2", &values2, &reference(&x2, gain2))
}
