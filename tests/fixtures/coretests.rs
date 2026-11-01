// Scalar operator and conversion cases transcribed from the Rust core tests, library/coretests/tests/num/ in
// rust-lang/rust at tag 1.98.0 (commit 88d9e12ae178fab0fb5cc050a94da85685d449ea). Copyright The Rust Project
// Developers; the Rust sources are licensed MIT OR Apache-2.0, as is this repository. Only the operations, the
// operand values and the expected results are taken from the tests. The kernels are written for the #[gpu]
// dialect and no test function or comment is copied. Where the source states no literal (it compares two
// spellings, or two expressions), the expected value was derived by hand and docs/development/CORETESTS.md
// marks the row "derived".
//
// Roadmap M5, slice m5.d1; the case table, the acceptance sets and the exclusions are in
// docs/development/CORETESTS.md, and crates/gust/tests/coretests_twin.rs runs the CPU twin of every kernel
// against the expected literals. The GPU half is slice m5.d2.
//
// Every kernel reads its operands from buffers, so the compiler cannot fold the expressions, and one invocation
// (global id 0) writes one result per slot of `out`. The ids in the comments are the CT-nn ids of the table.
// A comparison writes 1 for true and 0 for false. All values are inside the defined domain of
// docs/SEMANTICS.md section 3.
#[gust::gpu]
mod coretests {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn ints(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<int>,
        n: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<int>,
    ) {
        let i: uint = id.x;
        if i == 0u32 {
            let zl: int = a[0];
            let zr: int = a[1];
            let one: int = a[2];
            let ten: int = a[3];
            let two: int = a[4];
            let hi: int = a[5];
            let lo: int = a[6];
            let pat: int = a[7];
            let m12: int = a[8];
            let m10: int = a[9];
            let m7: int = a[10];
            let m14: int = a[11];
            let m3: int = a[12];
            let sh: uint = n[0];
            // CT-01: i32 -0 (unary minus)
            out[0] = -zl;
            // CT-02: i32 0 + 0, binary and compound-assignment spelling
            out[1] = zl + zr;
            // CT-02: second spelling or side
            let mut t2: int = zl;
            t2 += zr;
            out[2] = t2;
            // CT-03: i32 0 - 0, binary and compound-assignment spelling
            out[3] = zl - zr;
            // CT-03: second spelling or side
            let mut t4: int = zl;
            t4 -= zr;
            out[4] = t4;
            // CT-04: i32 0 * 0, binary and compound-assignment spelling
            out[5] = zl * zr;
            // CT-04: second spelling or side
            let mut t6: int = zl;
            t6 *= zr;
            out[6] = t6;
            // CT-05: i32 0 / 1, binary and compound-assignment spelling
            out[7] = zl / one;
            // CT-05: second spelling or side
            let mut t8: int = zl;
            t8 /= one;
            out[8] = t8;
            // CT-06: i32 0 % 1, binary and compound-assignment spelling
            out[9] = zl % one;
            // CT-06: second spelling or side
            let mut t10: int = zl;
            t10 %= one;
            out[10] = t10;
            // CT-07: i32 0 & 0, binary and compound-assignment spelling
            out[11] = zl & zr;
            // CT-07: second spelling or side
            let mut t12: int = zl;
            t12 &= zr;
            out[12] = t12;
            // CT-08: i32 0 | 0, binary and compound-assignment spelling
            out[13] = zl | zr;
            // CT-08: second spelling or side
            let mut t14: int = zl;
            t14 |= zr;
            out[14] = t14;
            // CT-09: i32 0 ^ 0, binary and compound-assignment spelling
            out[15] = zl ^ zr;
            // CT-09: second spelling or side
            let mut t16: int = zl;
            t16 ^= zr;
            out[16] = t16;
            // CT-10: i32 0 << 0, binary and compound-assignment spelling
            out[17] = zl << sh;
            // CT-10: second spelling or side
            let mut t18: int = zl;
            t18 <<= sh;
            out[18] = t18;
            // CT-11: i32 0 >> 0, binary and compound-assignment spelling
            out[19] = zl >> sh;
            // CT-11: second spelling or side
            let mut t20: int = zl;
            t20 >>= sh;
            out[20] = t20;
            // CT-31: i32 10 + 2 (the source asserts method == operator; value derived)
            out[21] = ten + two;
            // CT-32: i32 10 - 2 (the source asserts method == operator; value derived)
            out[22] = ten - two;
            // CT-33: i32 10 * 2 (the source asserts method == operator; value derived)
            out[23] = ten * two;
            // CT-34: i32 10 / 2 (the source asserts method == operator; value derived)
            out[24] = ten / two;
            // CT-35: i32 10 % 2 (the source asserts method == operator; value derived)
            out[25] = ten % two;
            // CT-45: i32 a * (a - 1)
            out[26] = ten * (ten - 1i32);
            // CT-46: i32 a - 10
            out[27] = ten - 10i32;
            // CT-47: i32 a / 10
            out[28] = ten / 10i32;
            // CT-48: i32 a - 20
            out[29] = ten - 20i32;
            // CT-49: i32 a << 10
            out[30] = ten << 10u32;
            // CT-50: i32 a << 16
            out[31] = ten << 16u32;
            // CT-51: i32 a * 16
            out[32] = ten * 16i32;
            // CT-52: i32 a * a * a
            out[33] = ten * ten * ten;
            // CT-53: i32 a * a * a * a
            out[34] = ten * ten * ten * ten;
            // CT-54: i32 a * a / a * a
            out[35] = ten * ten / ten * ten;
            // CT-55: i32 a * (a - 1) << (2 + a as usize), shift amount as u32
            out[36] = (ten * (ten - 1i32)) << (2u32 + (ten as uint));
            // CT-56: i32 b + 1 - 1 (b = 0x10101010)
            out[37] = pat + 1i32 - 1i32;
            // CT-57: i32 b & b << 1
            out[38] = pat & (pat << 1u32);
            // CT-58: i32 b | b << 1
            out[39] = pat | (pat << 1u32);
            // CT-59: i32 MAX > 0
            if hi > 0i32 {
                out[40] = 1i32;
            } else {
                out[40] = 0i32;
            }
            // CT-60: i32 MIN <= 0
            if lo <= 0i32 {
                out[41] = 1i32;
            } else {
                out[41] = 0i32;
            }
            // CT-61: i32 MIN + MAX + 1 (= -1 + 1, no overflow)
            out[42] = lo + hi + 1i32;
            // CT-62: i32 0b1100 | 0b1010 (source: .bitor)
            out[43] = m12 | m10;
            // CT-63: i32 0b1100 & 0b1010 (source: .bitand)
            out[44] = m12 & m10;
            // CT-64: i32 0b1100 ^ 0b1010 (source: .bitxor)
            out[45] = m12 ^ m10;
            // CT-65: i32 0b0111 << 1 (source: .shl)
            out[46] = m7 << 1u32;
            // CT-66: i32 0b1110 >> 1 (source: .shr)
            out[47] = m14 >> 1u32;
            // CT-67: i32 -(0b11) - 1 and !0b11 (source: .not), both sides
            out[48] = -m3 - 1i32;
            // CT-67: second spelling or side
            out[49] = !m3;
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn uints(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let i: uint = id.x;
        if i == 0u32 {
            let zl: uint = a[0];
            let zr: uint = a[1];
            let one: uint = a[2];
            let ten: uint = a[3];
            let two: uint = a[4];
            let hi: uint = a[5];
            let lo: uint = a[6];
            let m12: uint = a[7];
            let m10: uint = a[8];
            let m7: uint = a[9];
            let m14: uint = a[10];
            let m11: uint = a[11];
            let sh: uint = a[12];
            // CT-12: u32 0 + 0, binary and compound-assignment spelling
            out[0] = zl + zr;
            // CT-12: second spelling or side
            let mut t1: uint = zl;
            t1 += zr;
            out[1] = t1;
            // CT-13: u32 0 - 0, binary and compound-assignment spelling
            out[2] = zl - zr;
            // CT-13: second spelling or side
            let mut t3: uint = zl;
            t3 -= zr;
            out[3] = t3;
            // CT-14: u32 0 * 0, binary and compound-assignment spelling
            out[4] = zl * zr;
            // CT-14: second spelling or side
            let mut t5: uint = zl;
            t5 *= zr;
            out[5] = t5;
            // CT-15: u32 0 / 1, binary and compound-assignment spelling
            out[6] = zl / one;
            // CT-15: second spelling or side
            let mut t7: uint = zl;
            t7 /= one;
            out[7] = t7;
            // CT-16: u32 0 % 1, binary and compound-assignment spelling
            out[8] = zl % one;
            // CT-16: second spelling or side
            let mut t9: uint = zl;
            t9 %= one;
            out[9] = t9;
            // CT-17: u32 0 & 0, binary and compound-assignment spelling
            out[10] = zl & zr;
            // CT-17: second spelling or side
            let mut t11: uint = zl;
            t11 &= zr;
            out[11] = t11;
            // CT-18: u32 0 | 0, binary and compound-assignment spelling
            out[12] = zl | zr;
            // CT-18: second spelling or side
            let mut t13: uint = zl;
            t13 |= zr;
            out[13] = t13;
            // CT-19: u32 0 ^ 0, binary and compound-assignment spelling
            out[14] = zl ^ zr;
            // CT-19: second spelling or side
            let mut t15: uint = zl;
            t15 ^= zr;
            out[15] = t15;
            // CT-20: u32 0 << 0, binary and compound-assignment spelling
            out[16] = zl << sh;
            // CT-20: second spelling or side
            let mut t17: uint = zl;
            t17 <<= sh;
            out[17] = t17;
            // CT-21: u32 0 >> 0, binary and compound-assignment spelling
            out[18] = zl >> sh;
            // CT-21: second spelling or side
            let mut t19: uint = zl;
            t19 >>= sh;
            out[19] = t19;
            // CT-36: u32 10 + 2 (the source asserts method == operator; value derived)
            out[20] = ten + two;
            // CT-37: u32 10 - 2 (the source asserts method == operator; value derived)
            out[21] = ten - two;
            // CT-38: u32 10 * 2 (the source asserts method == operator; value derived)
            out[22] = ten * two;
            // CT-39: u32 10 / 2 (the source asserts method == operator; value derived)
            out[23] = ten / two;
            // CT-40: u32 10 % 2 (the source asserts method == operator; value derived)
            out[24] = ten % two;
            // CT-68: u32 MAX > 0
            if hi > 0u32 {
                out[25] = 1u32;
            } else {
                out[25] = 0u32;
            }
            // CT-69: u32 MIN <= 0 (always true; the including test allows clippy::absurd_extreme_comparisons)
            if lo <= 0u32 {
                out[26] = 1u32;
            } else {
                out[26] = 0u32;
            }
            // CT-70: u32 0b1100 | 0b1010 (source: .bitor)
            out[27] = m12 | m10;
            // CT-71: u32 0b1100 & 0b1010 (source: .bitand)
            out[28] = m12 & m10;
            // CT-72: u32 0b1100 ^ 0b1010 (source: .bitxor)
            out[29] = m12 ^ m10;
            // CT-73: u32 0b0111 << 1 (source: .shl)
            out[30] = m7 << 1u32;
            // CT-74: u32 0b1110 >> 1 (source: .shr)
            out[31] = m14 >> 1u32;
            // CT-75: u32 MAX - 0b1011 and !0b1011 (source: .not), both sides
            out[32] = hi - m11;
            // CT-75: second spelling or side
            out[33] = !m11;
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn floats(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i: uint = id.x;
        if i == 0u32 {
            let zl: float = a[0];
            let zr: float = a[1];
            let one: float = a[2];
            let ten: float = a[3];
            let two: float = a[4];
            let nz: float = a[5];
            // CT-22: f32 -0.0 (unary minus), compared with ==
            out[0] = -zl;
            // CT-23: f32 0 + 0, binary and compound-assignment spelling
            out[1] = zl + zr;
            // CT-23: second spelling or side
            let mut t2: float = zl;
            t2 += zr;
            out[2] = t2;
            // CT-24: f32 0 - 0, binary and compound-assignment spelling
            out[3] = zl - zr;
            // CT-24: second spelling or side
            let mut t4: float = zl;
            t4 -= zr;
            out[4] = t4;
            // CT-25: f32 0 * 0, binary and compound-assignment spelling
            out[5] = zl * zr;
            // CT-25: second spelling or side
            let mut t6: float = zl;
            t6 *= zr;
            out[6] = t6;
            // CT-26: f32 0 / 1, binary and compound-assignment spelling
            out[7] = zl / one;
            // CT-26: second spelling or side
            let mut t8: float = zl;
            t8 /= one;
            out[8] = t8;
            // CT-41: f32 10.0 + 2.0 (the source asserts method == operator; value derived)
            out[9] = ten + two;
            // CT-42: f32 10.0 - 2.0 (the source asserts method == operator; value derived)
            out[10] = ten - two;
            // CT-43: f32 10.0 * 2.0 (the source asserts method == operator; value derived)
            out[11] = ten * two;
            // CT-44: f32 10.0 / 2.0 (the source asserts method == operator; value derived)
            out[12] = ten / two;
            // CT-76: f32 0.0 == -0.0 (the bit-equality at 559 is excluded)
            if 0.0f32 == nz {
                out[13] = 1.0f32;
            } else {
                out[13] = 0.0f32;
            }
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn bools(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let i: uint = id.x;
        if i == 0u32 {
            let f1: bool = a[0] != 0u32;
            let f2: bool = a[1] != 0u32;
            let t1: bool = a[2] != 0u32;
            // CT-27: bool !true
            if !t1 {
                out[0] = 1u32;
            } else {
                out[0] = 0u32;
            }
            // CT-28: bool false & false, binary and compound-assignment spelling
            if f1 & f2 {
                out[1] = 1u32;
            } else {
                out[1] = 0u32;
            }
            // CT-28: second spelling or side
            let mut t2: bool = f1;
            t2 &= f2;
            if t2 {
                out[2] = 1u32;
            } else {
                out[2] = 0u32;
            }
            // CT-29: bool false | false, binary and compound-assignment spelling
            if f1 | f2 {
                out[3] = 1u32;
            } else {
                out[3] = 0u32;
            }
            // CT-29: second spelling or side
            let mut t4: bool = f1;
            t4 |= f2;
            if t4 {
                out[4] = 1u32;
            } else {
                out[4] = 0u32;
            }
            // CT-30: bool false ^ false, binary and compound-assignment spelling
            if f1 ^ f2 {
                out[5] = 1u32;
            } else {
                out[5] = 0u32;
            }
            // CT-30: second spelling or side
            let mut t6: bool = f1;
            t6 ^= f2;
            if t6 {
                out[6] = 1u32;
            } else {
                out[6] = 0u32;
            }
        }
    }

    // One element per case: the integer, its conversion to f32, and that float converted back to i32.
    // a[0] = 16777214: CT-77
    // a[1] = 16777215: CT-78
    // a[2] = 16777216: CT-79
    // a[3] = 16777217: CT-80
    // a[4] = -16777214: CT-81
    // a[5] = -16777215: CT-82
    // a[6] = -16777216: CT-83
    // a[7] = -16777217: CT-84
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn casts(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<int>,
        mut f: RWStructuredBuffer<float>,
        mut r: RWStructuredBuffer<int>,
    ) {
        let i: uint = id.x;
        if i < f.len() {
            f[i] = a[i] as float;
            r[i] = (a[i] as float) as int;
        }
    }
}
