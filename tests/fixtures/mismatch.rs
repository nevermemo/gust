#[gust::gpu]
mod mismatch {
    // Known-mismatch ledger kernels (roadmap M3, m3.a1, docs/SEMANTICS.md). Every kernel
    // reads its operands from buffers, so the validator cannot reject them and the
    // compiler cannot fold them, and writes one result per element. `op` selects the
    // operation inside one kernel so a single dispatch covers a whole table of cases.
    // Reference: crates/gust-wgpu/tests/mismatch.rs pins the GPU result and the plain Rust
    // result of every case. These kernels are probes of undefined or divergent domains,
    // not examples of portable code.

    // KM-01, KM-02, KM-03 (unsigned): op 0 is `a / b`, anything else `a % b`.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn udivide(
        id: SV_DispatchThreadID,
        op: StructuredBuffer<uint>,
        a: StructuredBuffer<uint>,
        b: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let i: uint = id.x;
        if i < out.len() {
            if op[i] == 0u32 {
                out[i] = a[i] / b[i];
            } else {
                out[i] = a[i] % b[i];
            }
        }
    }

    // KM-01, KM-02, KM-03 (signed): op 0 is `a / b`, anything else `a % b`.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn idivide(
        id: SV_DispatchThreadID,
        op: StructuredBuffer<uint>,
        a: StructuredBuffer<int>,
        b: StructuredBuffer<int>,
        mut out: RWStructuredBuffer<int>,
    ) {
        let i: uint = id.x;
        if i < out.len() {
            if op[i] == 0u32 {
                out[i] = a[i] / b[i];
            } else {
                out[i] = a[i] % b[i];
            }
        }
    }

    // KM-04 (unsigned): op 0 is `a << n`, anything else `a >> n`.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn ushift(
        id: SV_DispatchThreadID,
        op: StructuredBuffer<uint>,
        a: StructuredBuffer<uint>,
        n: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let i: uint = id.x;
        if i < out.len() {
            if op[i] == 0u32 {
                out[i] = a[i] << n[i];
            } else {
                out[i] = a[i] >> n[i];
            }
        }
    }

    // KM-04 (signed): op 0 is `a << n`, anything else the arithmetic shift `a >> n`.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn ishift(
        id: SV_DispatchThreadID,
        op: StructuredBuffer<uint>,
        a: StructuredBuffer<int>,
        n: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<int>,
    ) {
        let i: uint = id.x;
        if i < out.len() {
            if op[i] == 0u32 {
                out[i] = a[i] << n[i];
            } else {
                out[i] = a[i] >> n[i];
            }
        }
    }

    // KM-10 (unsigned): op 0 is `a + b`, 1 is `a - b`, anything else `a * b`.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn uarith(
        id: SV_DispatchThreadID,
        op: StructuredBuffer<uint>,
        a: StructuredBuffer<uint>,
        b: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let i: uint = id.x;
        if i < out.len() {
            if op[i] == 0u32 {
                out[i] = a[i] + b[i];
            } else if op[i] == 1u32 {
                out[i] = a[i] - b[i];
            } else {
                out[i] = a[i] * b[i];
            }
        }
    }

    // KM-10 (signed): op 0 is `a + b`, 1 is `a - b`, 2 is `a * b`, anything else `-a`.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn iarith(
        id: SV_DispatchThreadID,
        op: StructuredBuffer<uint>,
        a: StructuredBuffer<int>,
        b: StructuredBuffer<int>,
        mut out: RWStructuredBuffer<int>,
    ) {
        let i: uint = id.x;
        if i < out.len() {
            if op[i] == 0u32 {
                out[i] = a[i] + b[i];
            } else if op[i] == 1u32 {
                out[i] = a[i] - b[i];
            } else if op[i] == 2u32 {
                out[i] = a[i] * b[i];
            } else {
                out[i] = -a[i];
            }
        }
    }

    // The wrapping forms of KM-10 (slice m7.w), inside the domain for every input: op 0 is `a.wrapping_add(b)`,
    // 1 is `a.wrapping_sub(b)`, 2 is `a.wrapping_mul(b)`, anything else `a.wrapping_neg()`.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn uwrap(
        id: SV_DispatchThreadID,
        op: StructuredBuffer<uint>,
        a: StructuredBuffer<uint>,
        b: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let i: uint = id.x;
        if i < out.len() {
            if op[i] == 0u32 {
                out[i] = a[i].wrapping_add(b[i]);
            } else if op[i] == 1u32 {
                out[i] = a[i].wrapping_sub(b[i]);
            } else if op[i] == 2u32 {
                out[i] = a[i].wrapping_mul(b[i]);
            } else {
                out[i] = a[i].wrapping_neg();
            }
        }
    }

    // As `uwrap` for `i32`.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn iwrap(
        id: SV_DispatchThreadID,
        op: StructuredBuffer<uint>,
        a: StructuredBuffer<int>,
        b: StructuredBuffer<int>,
        mut out: RWStructuredBuffer<int>,
    ) {
        let i: uint = id.x;
        if i < out.len() {
            if op[i] == 0u32 {
                out[i] = a[i].wrapping_add(b[i]);
            } else if op[i] == 1u32 {
                out[i] = a[i].wrapping_sub(b[i]);
            } else if op[i] == 2u32 {
                out[i] = a[i].wrapping_mul(b[i]);
            } else {
                out[i] = a[i].wrapping_neg();
            }
        }
    }

    // KM-05: `f as i32`.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn cast_int(
        id: SV_DispatchThreadID,
        f: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<int>,
    ) {
        let i: uint = id.x;
        if i < out.len() {
            out[i] = f[i] as int;
        }
    }

    // KM-06: `f as u32`.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn cast_uint(
        id: SV_DispatchThreadID,
        f: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let i: uint = id.x;
        if i < out.len() {
            out[i] = f[i] as uint;
        }
    }

    // KM-07, KM-08: op 0 is `a + b`, 1 is `a - b`, 2 is `a * b`, anything else `a / b`.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn float_op(
        id: SV_DispatchThreadID,
        op: StructuredBuffer<uint>,
        a: StructuredBuffer<float>,
        b: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i: uint = id.x;
        if i < out.len() {
            if op[i] == 0u32 {
                out[i] = a[i] + b[i];
            } else if op[i] == 1u32 {
                out[i] = a[i] - b[i];
            } else if op[i] == 2u32 {
                out[i] = a[i] * b[i];
            } else {
                out[i] = a[i] / b[i];
            }
        }
    }

    // KM-07, KM-08: op 0 is `a == b`, 1 is `a != b`, 2 is `a < b`, 3 is `a > b`, anything
    // else `!(a < b)`. The negated form is the point: with a NaN operand it differs from `a >= b`.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn compare(op: uint, a: float, b: float) -> bool {
        if op == 0u32 {
            a == b
        } else if op == 1u32 {
            a != b
        } else if op == 2u32 {
            a < b
        } else if op == 3u32 {
            a > b
        } else {
            !(a < b)
        }
    }

    // KM-07, KM-08: the result of `compare` as 1 for true and 0 for false.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn float_cmp(
        id: SV_DispatchThreadID,
        op: StructuredBuffer<uint>,
        a: StructuredBuffer<float>,
        b: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let i: uint = id.x;
        if i < out.len() {
            if compare(op[i], a[i], b[i]) {
                out[i] = 1u32;
            } else {
                out[i] = 0u32;
            }
        }
    }

    // KM-11: one invocation reads `source[idx[0]]` and writes `sink[idx[0]]`; the host
    // passes an index equal to the length of `sink`. For the load row `source` has the same
    // length, so the load is out of bounds and happens first. For the store rows `source` is
    // longer, so the load is in bounds by construction and the store is the only access out of
    // bounds (crates/gust-wgpu/tests/mismatch.rs).
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn out_of_bounds(
        id: SV_DispatchThreadID,
        idx: StructuredBuffer<uint>,
        source: StructuredBuffer<uint>,
        mut sink: RWStructuredBuffer<uint>,
        mut seen: RWStructuredBuffer<uint>,
    ) {
        let i: uint = id.x;
        if i < seen.len() {
            let k: uint = idx[0u32];
            seen[i] = source[k];
            sink[k] = 77u32;
        }
    }

    // KM-09: every invocation exchanges the shared cell with its own id.
    #[kernel(workgroup_size(1, 1, 1))]
    pub fn contend_exchange(id: SV_DispatchThreadID, mut cell: RWStructuredBuffer<uint>) {
        atomic_exchange(&mut cell[0], id.x);
    }

    // KM-09: every invocation tries to replace the sentinel 999 with its own id plus one.
    #[kernel(workgroup_size(1, 1, 1))]
    pub fn contend_compare_exchange(
        id: SV_DispatchThreadID,
        mut cell: RWStructuredBuffer<uint>,
    ) {
        atomic_compare_exchange(&mut cell[0], 999u32, id.x + 1u32);
    }
}
