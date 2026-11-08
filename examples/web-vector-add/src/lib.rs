//! The vector-add kernel on a browser GPU (D45, plan M4 section 2.6): `wasm32-unknown-unknown` plus wgpu's `webgpu`
//! backend, with the kernel's committed WGSL artifact (`gust-artifacts/`, a byte copy of `tests/precompiled/kernels`'
//! artifacts and the one place an example commits artifacts, because a browser build cannot run Slang; m4.r's re-bless
//! must re-copy them, and an xtask test fails while the two sets differ). `cargo xtask check-web` builds it
//! under `GUST_PROFILE=web`, runs `wasm-bindgen --target web`, serves `index.html` with the result from a loopback
//! server, opens it in headless Chrome and reads the one line that [`run`] logs to the console:
//!
//! - `GUST_OK <vendor> <architecture> | selected <module.kernel=key:variant:n ...> | subgroups MissingCapability` when both forms of the kernel (the `grid` form, a compute pass, and the
//!   `pixel` form, a render pass whose fragments write the output buffer, D120) equal the host reference and each
//!   other to the last bit, at [`LENGTH`] and at [`TWO_DRAWS`] elements, where the two words are what the browser says
//!   about the adapter (`GPUAdapter.info`; wgpu's own `name` is empty there), and when the `WgpuJob` contract of the
//!   browser holds for each submission (D69, `check_job_contract`): `status` is `Pending` right after the submit (in
//!   the same synchronous run, so deterministic) and `Complete` after the awaited readback, `wait` and `wait_timeout`
//!   return `Error::WouldBlock`, and so does the generated `dispatch`; and when the resident `map_gpu` of a function
//!   (which submits and does not wait there) reads back, through `to_vec_async`, what the host computes; and when a device
//!   opened with the `subgroups` capability is refused with `Error::MissingCapability` (wgpu's WebGPU backend maps no
//!   subgroup feature, D142; M8, slice m8.e). The `selected` part names, for each kernel, the key of the artifact the build
//!   embedded, the variant it selected (`-` for the portable body) and how many capabilities that body requires
//!   ([`selected_kernels`]); `check-web` compares it with the committed manifest;
//! - `GUST_FAIL <reason>` on any error or panic.
//!
//! The whole crate is `cfg(target_arch = "wasm32")`: on the host it is empty.
#![cfg(target_arch = "wasm32")]

use gust::JobStatus;
use gust::capability::Capability;
use gust::gpu;
use gust_wgpu::{DeviceOptions, Error, GpuBufferAccess, HeadlessDevice, WgpuJob};
use js_sys::{Function, Object, Promise, Reflect};
use wasm_bindgen::{JsCast, JsValue, prelude::wasm_bindgen};
use wasm_bindgen_futures::JsFuture;

/// More than one workgroup (64) and not a multiple of it, so the grid form's bounds check matters; and less than the
/// pixel form's row width (8192), so the pixel form runs one partial row, one scissored draw.
const LENGTH: usize = 1000;

/// A whole row and a tail of five (8192 + 5): the pixel form's second row and both of its scissored draws (D127), and
/// the grid form's last workgroup partly used.
const TWO_DRAWS: usize = 8197;

/// What the pixel form's output holds before the call: no sum of the inputs equals it, so an element that no fragment
/// wrote cannot pass.
const UNWRITTEN: f32 = -1.0e9;

/// The element-wise map of the same job (D60), run by [`double_add::map_async`] after the vector add: it opens the
/// default device itself (in the browser through `gust_wgpu::default_device_async`) and awaits the result. Small
/// integers and halves, so its result is exact too.
#[gpu]
pub fn double_add(a: f32, b: f32) -> f32 {
    a * 2.0 + b
}

#[gpu]
mod vector_add {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn add(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<float>,
        b: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = a[i] + b[i];
        }
    }

    /// The same sum as a `pixel` kernel (mx.w): one fragment per element, scissored to the live elements by the host,
    /// so there is no bounds check. The output is a storage buffer written from the fragment stage.
    #[kernel(shape = pixel)]
    pub fn add_pixel(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<float>,
        b: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        out[i] = a[i] + b[i];
    }
}

/// Runs both forms of the kernel on the browser's GPU and returns the line it logged: `GUST_OK <vendor> <architecture>` or
/// `GUST_FAIL <reason>`. It never rejects; a panic is logged as `GUST_FAIL panic: ...` first.
#[wasm_bindgen]
pub async fn run() -> String {
    std::panic::set_hook(Box::new(|info| {
        log(&format!("GUST_FAIL panic: {}", one_line(&info.to_string())));
    }));
    let line = match vector_add_on_the_browser_gpu().await {
        Ok(label) => format!("GUST_OK {label}"),
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

/// The vendor and architecture words of the browser's adapter, or `unknown` for a word it does not give.
async fn browser_adapter_label() -> String {
    async fn words() -> Result<String, JsValue> {
        let navigator = Reflect::get(&js_sys::global(), &JsValue::from_str("navigator"))?;
        let gpu = Reflect::get(&navigator, &JsValue::from_str("gpu"))?;
        let request: Function =
            Reflect::get(&gpu, &JsValue::from_str("requestAdapter"))?.dyn_into()?;
        // wgpu does not expose the browser adapter it chose (its `AdapterInfo` has only an empty name here), so this asks
        // for one the way `HeadlessDevice::new_async` does, with `powerPreference: "high-performance"`. A second request
        // with the same options is expected to return the same adapter (not verified; one adapter was measured). The
        // label describes the setup; the exact result check is what proves the GPU ran.
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

/// The index of the first element whose bits differ, or of the first missing one when the lengths differ; `None` when
/// the two are equal bit for bit.
fn first_difference(left: &[f32], right: &[f32]) -> Option<usize> {
    left.iter()
        .zip(right)
        .position(|(left, right)| left.to_bits() != right.to_bits())
        .or_else(|| (left.len() != right.len()).then_some(left.len().min(right.len())))
}

async fn vector_add_on_the_browser_gpu() -> Result<String, String> {
    let device = HeadlessDevice::new_async(wgpu::Backends::BROWSER_WEBGPU)
        .await
        .map_err(|error| format!("no WebGPU device: {error}"))?;
    both_forms(&device, LENGTH).await?;
    both_forms(&device, TWO_DRAWS).await?;

    // The map path: `map_async` opens the default device (a second device, through `default_device_async`), uploads
    // the host slices, dispatches the generated kernel and awaits the read-back. Small integers and halves, so its
    // result is exact too.
    let host_a: Vec<f32> = (0..LENGTH).map(|i| i as f32).collect();
    let host_b: Vec<f32> = (0..LENGTH).map(|i| 2.0 * i as f32 + 0.5).collect();
    let doubled: Vec<f32> = host_a
        .iter()
        .zip(&host_b)
        .map(|(a, b)| a * 2.0 + b)
        .collect();
    let mapped = double_add::map_async(&host_a, &host_b)
        .await
        .map_err(|error| format!("map_async failed: {error}"))?;
    if mapped != doubled {
        let first = mapped
            .iter()
            .zip(&doubled)
            .position(|(got, want)| got != want)
            .unwrap_or(mapped.len().min(doubled.len()));
        return Err(format!(
            "map_async result differs from the host reference: {} of {} values, first at index {first}",
            mapped.len(),
            doubled.len()
        ));
    }

    // The resident form: `map_gpu` runs on the default device that `map_async` has just opened, submits without waiting
    // on wasm32 (D69) and leaves the result on the device; `to_vec_async` reads it back.
    let resident = double_add::map_gpu(&host_a, &host_b)
        .map_err(|error| format!("map_gpu failed: {error}"))?;
    let resident_host = resident
        .to_vec_async()
        .await
        .map_err(|error| format!("map_gpu read-back failed: {error}"))?;
    if let Some(first) = first_difference(&resident_host, &doubled) {
        return Err(format!(
            "map_gpu result differs from the host reference: {} of {} values, first at index {first}",
            resident_host.len(),
            doubled.len()
        ));
    }
    subgroups_are_refused().await?;
    Ok(format!(
        "{} | selected {} | subgroups MissingCapability",
        browser_adapter_label().await,
        selected_kernels()
    ))
}

/// The body this build selected for each kernel, as `check-web` reads it: `<module>.<kernel>=<key>:<variant>:<n>` with the
/// key of the artifact the descriptor embeds (`none` when it embeds none), `-` for the portable body's empty variant
/// name, and `n` the number of capabilities the body requires. `check-web` compares each triple with the committed
/// manifest (M8, slice m8.e): the key must be one the manifest lists for that kernel, and a build for the browser may
/// select no body that needs a capability.
fn selected_kernels() -> String {
    let modules = [vector_add::MODULE_DESCRIPTOR, double_add::MODULE_DESCRIPTOR];
    modules
        .iter()
        .flat_map(|module| module.kernels.iter())
        .map(|kernel| {
            format!(
                "{}.{}={}:{}:{}",
                kernel.module,
                kernel.name,
                kernel
                    .artifacts
                    .first()
                    .map_or("none", |artifact| artifact.key),
                if kernel.variant.is_empty() {
                    "-"
                } else {
                    kernel.variant
                },
                kernel.requires.len()
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Opening a device with the `subgroups` capability must fail with `Error::MissingCapability` in the browser: wgpu 30's
/// WebGPU backend maps no subgroup feature (D142), so the request is refused with the typed error and no device exists.
async fn subgroups_are_refused() -> Result<(), String> {
    let options = DeviceOptions::default().with_capabilities(&[Capability::Subgroups]);
    match HeadlessDevice::new_async_with(options).await {
        Err(Error::MissingCapability {
            capability: Capability::Subgroups,
            ..
        }) => Ok(()),
        Err(other) => Err(format!(
            "opening a device with the subgroups capability must be MissingCapability in the browser, got {other}"
        )),
        Ok(_) => Err(
            "opening a device with the subgroups capability must fail in the browser, but a device opened".to_owned(),
        ),
    }
}

/// The browser contract of `WgpuJob` (D69), checked after the job's awaited readback has resolved: `status` is
/// `Complete` (the `on_submitted_work_done` callback has run), and `wait` and `wait_timeout` return
/// `Error::WouldBlock` rather than `Ok`. `before` is the status read right after the submit and is required to be
/// `Pending`: nothing has awaited or returned to the event loop between the submit and that read, so the
/// `on_submitted_work_done` callback cannot have run. That is not a timing guess; if Chrome ever shows `Complete`
/// there, the contract is broken.
fn check_job_contract(
    what: &str,
    job: &WgpuJob<'_>,
    before: Result<JobStatus, Error>,
) -> Result<(), String> {
    if !matches!(before, Ok(JobStatus::Pending)) {
        return Err(format!(
            "{what}: the job's status right after the submit must be Pending, got {before:?}"
        ));
    }
    match job.status() {
        Ok(JobStatus::Complete) => {}
        other => {
            return Err(format!(
                "{what}: the job's status after the awaited readback must be Complete, got {other:?}"
            ));
        }
    }
    if !matches!(job.wait(), Err(Error::WouldBlock(_))) {
        return Err(format!(
            "{what}: WgpuJob::wait must return Error::WouldBlock in the browser"
        ));
    }
    if !matches!(
        job.wait_timeout(std::time::Duration::from_millis(1)),
        Err(Error::WouldBlock(_))
    ) {
        return Err(format!(
            "{what}: WgpuJob::wait_timeout must return Error::WouldBlock in the browser"
        ));
    }
    Ok(())
}

/// Both forms of the vector add over `length` elements: the `grid` form against the host reference, then the `pixel` form
/// against the host reference and the grid result, all bit for bit.
async fn both_forms(device: &HeadlessDevice, length: usize) -> Result<(), String> {
    // Small integers and halves: every sum is exact in f32, so the comparison below is exact.
    let a: Vec<f32> = (0..length).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..length).map(|i| 2.0 * i as f32 + 0.5).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(left, right)| left + right).collect();

    let a = device.create_f32_buffer("web a", &a, GpuBufferAccess::ReadOnly);
    let b = device.create_f32_buffer("web b", &b, GpuBufferAccess::ReadOnly);
    let out = device.create_f32_buffer("web out", &vec![0.0; length], GpuBufferAccess::ReadWrite);
    // The job is not waited on: in the browser `WgpuJob::wait` returns an error (see its documentation). The awaited
    // readback below is what resolves when the dispatch and the copy are done.
    let job = vector_add::add::submit(device, length as u32, &a, &b, &out)
        .map_err(|error| format!("n = {length}: dispatch failed: {error}"))?;
    // Nothing has yielded to the event loop since the submit, so the callback cannot have run: `Pending` (required).
    let before = job.status();
    let result = device
        .read_typed_buffer_async(&out)
        .await
        .map_err(|error| format!("n = {length}: read-back failed: {error}"))?;
    check_job_contract(&format!("n = {length}: grid"), &job, before)?;
    let blocking = vector_add::add::dispatch(device, length as u32, &a, &b, &out);
    if !matches!(blocking, Err(Error::WouldBlock(_))) {
        return Err(format!(
            "n = {length}: the generated dispatch must return Error::WouldBlock in the browser, got {blocking:?}"
        ));
    }
    if let Some(first) = first_difference(&result, &expected) {
        return Err(format!(
            "n = {length}: grid result differs from the host reference: {} of {} values, first at index {first}",
            result.len(),
            expected.len()
        ));
    }

    // The `pixel` form of the same kernel: a render pass over a full-screen triangle, scissored to the `length` live
    // elements (one draw for a partial row, two for whole rows and a tail), whose fragments write the storage buffer
    // (D120, D127). The output starts as [`UNWRITTEN`].
    //
    // The browser reports no downlevel limits to wgpu (wgpu 30's web backend answers "every flag set"), so gust-compute-wgpu
    // cannot refuse a pixel kernel here with `ShapeUnsupported`. An adapter that cannot write storage from a fragment
    // fails when the pipeline is created, as a WebGPU validation error in the browser's console, and the output keeps
    // its fill: the page then logs the "no fragment wrote" reason below, not a refusal.
    let pixel_out = device.create_f32_buffer(
        "web pixel out",
        &vec![UNWRITTEN; length],
        GpuBufferAccess::ReadWrite,
    );
    let pixel_job = vector_add::add_pixel::submit(device, length as u32, &a, &b, &pixel_out)
        .map_err(|error| format!("n = {length}: pixel dispatch failed: {error}"))?;
    let pixel_before = pixel_job.status();
    let pixel_result = device
        .read_typed_buffer_async(&pixel_out)
        .await
        .map_err(|error| format!("n = {length}: pixel read-back failed: {error}"))?;
    check_job_contract(&format!("n = {length}: pixel"), &pixel_job, pixel_before)?;
    if !pixel_result.is_empty()
        && pixel_result
            .iter()
            .all(|value| value.to_bits() == UNWRITTEN.to_bits())
    {
        return Err(format!(
            "n = {length}: no fragment wrote the output: the pixel pass did not run (fragment-stage storage writes? see the console's WebGPU errors)"
        ));
    }
    if let Some(first) = first_difference(&pixel_result, &expected) {
        return Err(format!(
            "n = {length}: pixel result differs from the host reference: {} of {} values, first at index {first}",
            pixel_result.len(),
            expected.len()
        ));
    }
    if let Some(first) = first_difference(&pixel_result, &result) {
        return Err(format!(
            "n = {length}: pixel result differs from the grid result: {} of {} values, first at index {first}",
            pixel_result.len(),
            result.len()
        ));
    }
    Ok(())
}
