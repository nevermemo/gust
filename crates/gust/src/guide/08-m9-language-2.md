```

The input is longer than the invocation count (16 symbols per record), so the host passes it as `Independent(&text)`
([chapter 4][ch4]). The gallery entry's tests run this kernel on the CPU twin against a host tokenizer written in plain
Rust, and the program runs it on the GPU and compares the same way.

## Limits worth knowing

- **The array driver defect (D168, D169).** On the NVIDIA RTX 5090 with driver 610.88, a function-scope array can read
  zeros after an element at a dynamic index is loaded and written back with a value the driver can prove unchanged
  (`a[k] -= 0`, `a[k] = a[k]`). The generated Slang, WGSL and SPIR-V are correct and an AMD Radeon 890M computes Rust's
  values, so GUST does not refuse the form. The default WGSL route stays exposed on that driver; the opt-in
  `spirv-route` feature did not reproduce it there. This was measured on one device and driver; section 7 of SEMANTICS
  has the full pattern.
- **Method order.** A `&mut self` call whose order of evaluation on the GPU would differ from Rust's is rejected
  (GUST-E0421).
- **Closures, tuples and enums have no host layout.** They exist inside a kernel; none can be a buffer element or a
  uniform field, so a result that leaves the kernel is a scalar, a vector, a matrix, an array or a struct of these.
- **Edge cases.** Where the GPU result may differ from the CPU twin's for an input at the edge of the defined domain,
  SEMANTICS lists it as a limitation and the project does not chase a match (D39).

[Chapter 7][`crate::guide::semantics_and_the_twin`] states the contract that all of this lives inside.
