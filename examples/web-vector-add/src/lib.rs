//! The vector-add kernel on a browser GPU (D45, plan M4 section 2.6): `wasm32-unknown-unknown` plus wgpu's `webgpu`
//! backend, with the kernel's committed WGSL artifact (`gust-artifacts/`, a byte copy of `tests/precompiled/kernels`'
//! artifacts and the one place an example commits artifacts, because a browser build cannot run Slang; m4.r's re-bless
//! must re-copy them, and an xtask test fails while the two sets differ). `cargo xtask check-web` builds it
//! under `GUST_PROFILE=web`, runs `wasm-bindgen --target web`, serves `index.html` with the result from a loopback
//! server, opens it in headless Chrome and reads the one line that [`run`] logs to the console:
//!
//! - `GUST_OK <vendor> <architecture>` when both forms of the kernel (the `grid` form, a compute pass, and the
//!   `pixel` form, a render pass whose fragments write the output buffer, D120) equal the host reference and each
//!   other to the last bit, where the two words are what the browser says about the adapter (`GPUAdapter.info`;
//!   wgpu's own `name` is empty there);
//! - `GUST_FAIL <reason>` on any error or panic.
//!
//! The whole crate is `cfg(target_arch = "wasm32")`: on the host it is empty.
#![cfg(target_arch = "wasm32")]

use gust::gpu;
use gust_wgpu::{GpuBufferAccess, HeadlessDevice};
use js_sys::{Function, Object, Promise, Reflect};
use wasm_bindgen::{JsCast, JsValue, prelude::wasm_bindgen};
use wasm_bindgen_futures::JsFuture;

/// More than one workgroup (64) and not a multiple of it, so the kernel's bounds check matters.
const LENGTH: usize = 1000;

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
    // Small integers and halves: every sum is exact in f32, so the comparison below is exact.
    let a: Vec<f32> = (0..LENGTH).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..LENGTH).map(|i| 2.0 * i as f32 + 0.5).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(left, right)| left + right).collect();
    // The host inputs and reference of the map path below (`a` and `b` become device buffers).
    let doubled: Vec<f32> = a.iter().zip(&b).map(|(a, b)| a * 2.0 + b).collect();
    let (host_a, host_b) = (a.clone(), b.clone());

    let device = HeadlessDevice::new_async(wgpu::Backends::BROWSER_WEBGPU)
        .await
        .map_err(|error| format!("no WebGPU device: {error}"))?;
    let a = device.create_f32_buffer("web a", &a, GpuBufferAccess::ReadOnly);
    let b = device.create_f32_buffer("web b", &b, GpuBufferAccess::ReadOnly);
    let out = device.create_f32_buffer("web out", &vec![0.0; LENGTH], GpuBufferAccess::ReadWrite);
    // The job is not waited on: in the browser `WgpuJob::wait` returns at once (see its documentation). The awaited
    // readback below is what resolves when the dispatch and the copy are done.
    let _job = vector_add::add::submit(&device, LENGTH as u32, &a, &b, &out)
        .map_err(|error| format!("dispatch failed: {error}"))?;
    let result = device
        .read_typed_buffer_async(&out)
        .await
        .map_err(|error| format!("read-back failed: {error}"))?;

    if let Some(first) = first_difference(&result, &expected) {
        return Err(format!(
            "grid result differs from the host reference: {} of {} values, first at index {first}",
            result.len(),
            expected.len()
        ));
    }

    // The `pixel` form of the same kernel: a render pass over a full-screen triangle, scissored to the LENGTH live
    // elements, whose fragments write the storage buffer (D120, D127). Its output starts as a value no sum equals,
    // so an element no fragment wrote cannot pass. A browser that cannot write storage from a fragment answers with
    // `ShapeUnsupported` here, and the reason is what the page logs.
    let pixel_out = device.create_f32_buffer(
        "web pixel out",
        &vec![-1.0e9; LENGTH],
        GpuBufferAccess::ReadWrite,
    );
    let _pixel_job = vector_add::add_pixel::submit(&device, LENGTH as u32, &a, &b, &pixel_out)
        .map_err(|error| format!("pixel dispatch failed: {error}"))?;
    let pixel_result = device
        .read_typed_buffer_async(&pixel_out)
        .await
        .map_err(|error| format!("pixel read-back failed: {error}"))?;
    if let Some(first) = first_difference(&pixel_result, &expected) {
        return Err(format!(
            "pixel result differs from the host reference: {} of {} values, first at index {first}",
            pixel_result.len(),
            expected.len()
        ));
    }
    if let Some(first) = first_difference(&pixel_result, &result) {
        return Err(format!(
            "pixel result differs from the grid result: {} of {} values, first at index {first}",
            pixel_result.len(),
            result.len()
        ));
    }

    // The map path: `map_async` opens the default device (a second device, through `default_device_async`), uploads
    // the host slices, dispatches the generated kernel and awaits the read-back.
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
    Ok(browser_adapter_label().await)
}
