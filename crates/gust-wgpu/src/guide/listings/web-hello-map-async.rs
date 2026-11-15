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
