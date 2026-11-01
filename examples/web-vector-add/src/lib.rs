//! The vector-add kernel on a browser GPU (D45, plan M4 section 2.6): `wasm32-unknown-unknown` plus wgpu's `webgpu`
//! backend, with the kernel's committed WGSL artifact (`gust-artifacts/`, a byte copy of `tests/precompiled/kernels`'
//! artifacts and the one place an example commits artifacts, because a browser build cannot run Slang; m4.r's re-bless
//! must re-copy them, and an xtask test fails while the two sets differ). `cargo xtask check-web` builds it
//! under `GUST_PROFILE=web`, runs `wasm-bindgen --target web`, serves `index.html` with the result from a loopback
//! server, opens it in headless Chrome and reads the one line that [`run`] logs to the console:
//!
//! - `GUST_OK <vendor> <architecture>` when the GPU result equals the host reference to the last bit, where the two
//!   words are what the browser says about the adapter (`GPUAdapter.info`; wgpu's own `name` is empty there);
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
}

/// Runs the kernel on the browser's GPU and returns the line it logged: `GUST_OK <vendor> <architecture>` or
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

async fn vector_add_on_the_browser_gpu() -> Result<String, String> {
    // Small integers and halves: every sum is exact in f32, so the comparison below is exact.
    let a: Vec<f32> = (0..LENGTH).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..LENGTH).map(|i| 2.0 * i as f32 + 0.5).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(left, right)| left + right).collect();

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

    if result != expected {
        let first = result
            .iter()
            .zip(&expected)
            .position(|(got, want)| got != want)
            .unwrap_or(result.len().min(expected.len()));
        return Err(format!(
            "GPU result differs from the host reference: {} of {} values, first at index {first}",
            result.len(),
            expected.len()
        ));
    }
    Ok(browser_adapter_label().await)
}
