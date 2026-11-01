//! Prints the adapter `HeadlessDevice` selected as one JSON line on stdout, for `cargo xtask` to
//! copy into `.ai/VALIDATION.json`. No adapter is a failure (exit 1, nothing on stdout): there is
//! no skip and no fallback.
//!
//! Only identifying fields are printed; none of them is a path.

use gust_wgpu::HeadlessDevice;

fn main() {
    let device = match HeadlessDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("adapter_info: no adapter: {error}");
            std::process::exit(1);
        }
    };
    println!("{}", adapter_json(device.adapter_info()));
}

/// The fields of `info` as a single-line JSON object.
fn adapter_json(info: &wgpu::AdapterInfo) -> String {
    format!(
        "{{\"name\":{},\"vendor\":{},\"device\":{},\"device_type\":{},\"backend\":{},\"driver\":{},\"driver_info\":{}}}",
        json_string(&info.name),
        info.vendor,
        info.device,
        json_string(&format!("{:?}", info.device_type)),
        json_string(&format!("{:?}", info.backend)),
        json_string(&info.driver),
        json_string(&info.driver_info),
    )
}

fn json_string(value: &str) -> String {
    let mut out = String::from("\"");
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
