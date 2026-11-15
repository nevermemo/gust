```

## The browser side

The page's own code is in a module that exists on `wasm32` only (`#[cfg(target_arch = "wasm32")] mod browser`), which
is why no host doctest can compile it, and why the block below is plain text and not a listing a test builds. The
function is the heart of it: it calls the awaiting map, compares the result with the reference bit for bit, and returns
the words that go on the console line. The helpers it names (`first_difference`, `selected_kernels` and
`browser_adapter_label`) are in the same module; the region is shown with its indentation inside `mod browser`.

```text
