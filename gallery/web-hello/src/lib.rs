//! web-hello (gallery G9): the hello-gust sum in a browser. One `#[gpu_fn]` map function, run on the browser's GPU
//! through `map_async` and checked, in the page, against a plain Rust host reference. `cargo xtask check-gallery`
//! builds the crate for `wasm32-unknown-unknown`, serves it from a loopback server, opens it in headless Chrome and
//! reads the one line the page's Rust code logs to the console (see the README). On the host the crate is a library
//! whose tests run the same kernel on the CPU twin.

// ANCHOR: imports
use gust::gpu_fn;
// ANCHOR_END: imports

// ANCHOR: kernel
/// The element-wise sum of two vectors. `#[gpu_fn]` on a plain scalar function keeps the function as written and adds a
/// module of the same name with the kernel and the calls that run it (`map`, `map_async`, ..).
#[gpu_fn]
pub fn add(a: f32, b: f32) -> f32 {
    a + b
}
// ANCHOR_END: kernel

// ANCHOR: reference
/// How many elements the page adds: more than one workgroup (64) and not a multiple of it.
pub const LENGTH: usize = 1000;

/// The inputs of `n` elements: small integers, so that every sum is exact in `f32` on any target.
pub fn inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
    let a = (0..n).map(|i| (i % 97) as f32).collect();
    let b = (0..n).map(|i| (i % 13) as f32 * 8.0).collect();
    (a, b)
}

/// The independent host reference: plain Rust over slices, which never goes through the kernel (D26, D42).
pub fn host_reference(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}
// ANCHOR_END: reference

// ANCHOR: browser
/// The browser side, compiled for `wasm32` only. The page's `index.html` calls [`run`] and the runner reads the console
/// line it logs: `GUST_OK <vendor> <architecture> | selected <kernels> | <what held>`, or `GUST_FAIL <reason>`.
#[cfg(target_arch = "wasm32")]
mod browser {
    use js_sys::{Function, Object, Promise, Reflect};
    use wasm_bindgen::{JsCast, JsValue, prelude::wasm_bindgen};
    use wasm_bindgen_futures::JsFuture;

    use super::{LENGTH, add, host_reference, inputs};

    /// The last part of the OK line. `cargo xtask check-gallery` requires exactly this text (a unit test of xtask finds it
    /// in this file), so a page that did not compare its result cannot be taken for one that did.
    const HELD: &str = "map_async exact against the host reference";

    /// Runs the kernel on the browser's GPU and returns the line it logged. It never rejects; a panic is logged as
    /// `GUST_FAIL panic: ...` first.
    #[wasm_bindgen]
    pub async fn run() -> String {
        std::panic::set_hook(Box::new(|info| {
            log(&format!("GUST_FAIL panic: {}", one_line(&info.to_string())));
        }));
        let line = match add_on_the_browser_gpu().await {
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

    // ANCHOR: map-async
    /// `map_async` opens the default device itself (in the browser through `default_device_async`), uploads the host
    /// slices, dispatches the generated kernel and awaits the read-back; nothing blocks, which a browser cannot allow.
    async fn add_on_the_browser_gpu() -> Result<String, String> {
        let (a, b) = inputs(LENGTH);
        let expected = host_reference(&a, &b);
        let result = add::map_async(&a, &b)
            .await
            .map_err(|error| format!("map_async failed: {error}"))?;
        if let Some(first) = first_difference(&result, &expected) {
            return Err(format!(
                "map_async result differs from the host reference: {} values against {} expected, first difference at index {first}",
                result.len(),
                expected.len()
            ));
        }
        Ok(format!(
            "{} | selected {} | {HELD}",
            browser_adapter_label().await,
            selected_kernels()
        ))
    }
    // ANCHOR_END: map-async

    /// The index of the first element whose bits differ, or of the first missing one when the lengths differ; `None` when
    /// the two are equal bit for bit.
    fn first_difference(left: &[f32], right: &[f32]) -> Option<usize> {
        left.iter()
            .zip(right)
            .position(|(left, right)| left.to_bits() != right.to_bits())
            .or_else(|| (left.len() != right.len()).then_some(left.len().min(right.len())))
    }

    /// The body this build selected for each kernel, as the runner reads it: `<module>.<kernel>=<key>:<variant>:<n>` with
    /// the key of the artifact the descriptor embeds (`none` when it embeds none), `-` for the portable body's empty
    /// variant name, and `n` the number of capabilities the body requires. The runner compares each with the committed
    /// `gust-artifacts/manifest.json`.
    fn selected_kernels() -> String {
        add::MODULE_DESCRIPTOR
            .kernels
            .iter()
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

    /// The vendor and architecture words of the browser's adapter, or `unknown` for a word it does not give. The label
    /// describes the setup (a second request with the same options is expected to return the adapter `map_async`
    /// used); the exact comparison above is what proves a GPU ran.
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
            let adapter =
                JsFuture::from(request.call1(&gpu, &options)?.dyn_into::<Promise>()?).await?;
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
}
// ANCHOR_END: browser

// ANCHOR: tests
#[cfg(test)]
mod tests {
    use gust::cpu::CpuDevice;

    use super::{add, host_reference, inputs};

    /// The kernel on the CPU twin (the macro emits the same `map_on` for `CpuDevice` under `cfg(test)`).
    fn on_twin(a: &[f32], b: &[f32]) -> Vec<f32> {
        add::map_on(&CpuDevice::new(), &a.to_vec(), &b.to_vec()).expect("the twin runs the map")
    }

    #[test]
    fn the_twin_matches_the_program_values() {
        let result = on_twin(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0]);
        assert_eq!(result, [11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn the_twin_matches_the_host_reference() {
        // Sizes below, at and above the workgroup size of 64, one that is not a multiple of it, and the page's own.
        for n in [1, 63, 64, 65, super::LENGTH] {
            let (a, b) = inputs(n);
            assert_eq!(on_twin(&a, &b), host_reference(&a, &b), "n = {n}");
        }
    }

    #[test]
    fn the_host_reference_is_independent_of_the_kernel() {
        // Hand-counted: the inputs are `i % 97` and `(i % 13) * 8`, so element 14 is 14 + 8 and element 100 is 3 + 72.
        let (a, b) = inputs(101);
        let expected = host_reference(&a, &b);
        assert_eq!(expected[0], 0.0);
        assert_eq!(expected[14], 22.0);
        assert_eq!(expected[100], 75.0);
    }
}
// ANCHOR_END: tests
