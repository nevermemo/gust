# Scalar coretests transcription

This page records which scalar integer and float cases of the Rust core tests GUST transcribes into `#[gpu]`
kernels, the literals they must produce, the sets the GPU may produce instead, and what was left out and why. It is
the record of roadmap slice m5.d1 (M5, "coretests on every target"). The kernels are `tests/fixtures/coretests.rs`;
`crates/gust/tests/coretests_twin.rs` runs the CPU twin of every kernel against the literals of the table below
(Unit tier, no adapter). The GPU half, the same kernels on a hardware adapter and on the software adapter with the
acceptance sets of the "Acceptance" column, is `crates/gust-wgpu/tests/coretests.rs` (slice m5.d2); section 7
records its runs.

The expected literal is the independent reference (D26, `docs/SEMANTICS.md`): it comes from the Rust sources or is
derived by hand, never from the twin or from a GPU. The twin is asserted exact in every row and is never GPU
evidence. The only GPU evidence on this page is section 7, the runs of slice m5.d2; the table and sections 1 to 6 are
unchanged by them.

## 1. Pin and procedure

Source: `rust-lang/rust`, tag `1.98.0` (the toolchain of CI), commit `88d9e12ae178fab0fb5cc050a94da85685d449ea`,
directory `library/coretests/tests/num/`. The values of `f32::MAX_EXACT_INTEGER` and `MIN_EXACT_INTEGER` come from
`library/core/src/num/f32.rs` of the same commit (lines 544 and 574: 2^24 - 1 and its negation). Line numbers in the
table are lines of those files at that commit.

The procedure (M5 plan, section 2.4 with Revision 1, R1-7):

1. Pin the tag and record the commit. Read the files of `tests/num/` that can hold scalar integer or float
   assertions: `ops.rs`, `mod.rs`, `int_macros.rs`, `uint_macros.rs`, `i32.rs`, `u32.rs`, `floats.rs`, `cast.rs` and,
   to confirm that they hold none, `midpoint.rs`, `wrapping.rs`, `int_log.rs`, `int_sqrt.rs`, `const_from.rs`,
   `clamp_magnitude.rs`, `nan.rs`, `carryless_mul.rs`, `float_iter_sum_identity.rs`. The downloads are data only;
   nothing from them is built or run.
2. Include an assertion when it can be written in the dialect today: the operators `+ - * / % & | ^ << >>`, unary `-`
   and `!`, comparisons and `as` casts among `i32`, `u32`, `f32`, with `bool` for the results of comparisons and for
   `!`, `&`, `|`, `^`. An `isize` or `usize` test is transcribed to `i32` or `u32` when every value fits. A method
   spelling of an operator (`.bitor(x)`, `.shl(n)`, `.not()`) is transcribed to the operator, and the row says so.
   `MIN` and `MAX` are not paths in the dialect, so they enter as input values.
3. Exclude, with the reason in section 5, every method test, every NaN or infinity case, every case outside the defined
   domain (`docs/SEMANTICS.md` section 3), float `%`, and every case that needs a type or construct the dialect lacks.
   A case that would need a construct the dialect lacks is excluded, never approximated.
4. Give each remaining operation an id `CT-nn`. A row is one operation on one type from one test. Where the source has
   a binary test and a compound-assignment test with the same literal (`ops.rs`), the kernel computes both spellings
   and the row asserts both slots; where the source compares two spellings or two expressions, the row asserts both.
   The reference forms of the source (`L op R`, `L op &R`, `&L op R`, `&L op &R`) are one operation in the dialect.
5. Record the expected literal. Its origin is `source` when the test states the literal (or the value of a named
   variable, or a truth value), and `derived` when the test compares two spellings or two expressions and the value
   follows from the operands: the `test_num` rows (`10 op 2`), the `!` rows of `int_macros.rs:31` and
   `uint_macros.rs:27`, and the casts of `floats.rs`, whose value follows from round-to-nearest-even and from the
   equalities the test asserts.
6. Record the acceptance set per R1-7 and the coverage triples (section 4).

## 2. Reading the table

- **Source**: file and line(s) at the pinned commit. For a macro instantiated per type, the instantiation line is
  given (`ops.rs:109,110`) and the macro bodies are `ops.rs:58-107`, `120-169` and `178-233`.
- **Kernel[slots]**: the kernel of `tests/fixtures/coretests.rs` and the `out[k]` slots that carry the row. The
  twin test and the GPU test read the same slots. A comparison stores 1 for true and 0 for false.
- **Expected**: the literal, or several distinct literals separated by `;` when a row asserts slots of different
  types (the casts: the `f32` value, then the `i32` round trip).
- **Acceptance**: the set of values the GPU may return. `exact` is bit equality for integers and `bool`, and `==` for
  floats (WGSL lets an implementation ignore the sign of a zero, 15.7.2). `GPU: 2.5 ULP set` is float `/`: every `f32`
  within 2.5 ULP of the literal, asserted as `gust::cpu::ulps(gpu, literal) <= 2`, because WGSL allows 2.5 ULP
  whatever the operands (`docs/SEMANTICS.md` section 4), so `10.0 / 2.0 == 5.0` is not a portable GPU assertion.
  `GPU: two-neighbour set` is an integer above 2^24 converted to `f32` (15.7.6): either neighbour of the integer, and
  the same set for the round trip back to `i32`. The twin is exact in every row, because it is Rust's own arithmetic.
- **Triples**: the (operator, type, operand class) triples the row exercises, each marked `[new]` or `[existing]`
  (section 4).
- **Origin**: `source` or `derived` (section 1, step 5).

Operand classes: `zero`; `small` (magnitude below 2^16, either sign); `pattern` (a bit mask or bit pattern: `0b1100`,
`0x10101010`); `boundary` (`MIN`, `MAX`, or within two of 2^24); `signed-zero` (-0.0); `true` and `false` for `bool`.

## 3. The table

All 32-bit. Inputs are the variables of the kernels (`zl`, `zr`, `one`, `ten`, `two`, `hi`, `lo`, `pat`, `m3` to
`m14`, `sh`, `nz`, `f1`, `f2`, `t1`, `x`); `hi` and `lo` are `MAX` and `MIN` of the type.

| ID | Source (at the pinned commit) | Operation | Inputs | Kernel[slots] | Expected | Acceptance | Triples | Origin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CT-01 | ops.rs:54 | i32 -0 (unary minus) | zl=0 | ints[0] | 0 | exact | neg i32 zero [new] | source |
| CT-02 | ops.rs:109,110 | i32 0 + 0, binary and compound-assignment spelling | zl=0, zr=0 | ints[1,2] | 0 | exact | + i32 zero [new] | source |
| CT-03 | ops.rs:111,112 | i32 0 - 0, binary and compound-assignment spelling | zl=0, zr=0 | ints[3,4] | 0 | exact | - i32 zero [new] | source |
| CT-04 | ops.rs:113,114 | i32 0 * 0, binary and compound-assignment spelling | zl=0, zr=0 | ints[5,6] | 0 | exact | * i32 zero [new] | source |
| CT-05 | ops.rs:115,116 | i32 0 / 1, binary and compound-assignment spelling | zl=0, one=1 | ints[7,8] | 0 | exact | / i32 zero [new] | source |
| CT-06 | ops.rs:117,118 | i32 0 % 1, binary and compound-assignment spelling | zl=0, one=1 | ints[9,10] | 0 | exact | % i32 zero [new] | source |
| CT-07 | ops.rs:171,172 | i32 0 & 0, binary and compound-assignment spelling | zl=0, zr=0 | ints[11,12] | 0 | exact | & i32 zero [new] | source |
| CT-08 | ops.rs:173,174 | i32 0 \| 0, binary and compound-assignment spelling | zl=0, zr=0 | ints[13,14] | 0 | exact | \| i32 zero [new] | source |
| CT-09 | ops.rs:175,176 | i32 0 ^ 0, binary and compound-assignment spelling | zl=0, zr=0 | ints[15,16] | 0 | exact | ^ i32 zero [new] | source |
| CT-10 | ops.rs:235,236 | i32 0 << 0, binary and compound-assignment spelling | zl=0, sh=0 | ints[17,18] | 0 | exact | << i32 zero [new] | source |
| CT-11 | ops.rs:237,238 | i32 0 >> 0, binary and compound-assignment spelling | zl=0, sh=0 | ints[19,20] | 0 | exact | >> i32 zero [new] | source |
| CT-12 | ops.rs:109,110 | u32 0 + 0, binary and compound-assignment spelling | zl=0, zr=0 | uints[0,1] | 0 | exact | + u32 zero [new] | source |
| CT-13 | ops.rs:111,112 | u32 0 - 0, binary and compound-assignment spelling | zl=0, zr=0 | uints[2,3] | 0 | exact | - u32 zero [new] | source |
| CT-14 | ops.rs:113,114 | u32 0 * 0, binary and compound-assignment spelling | zl=0, zr=0 | uints[4,5] | 0 | exact | * u32 zero [new] | source |
| CT-15 | ops.rs:115,116 | u32 0 / 1, binary and compound-assignment spelling | zl=0, one=1 | uints[6,7] | 0 | exact | / u32 zero [new] | source |
| CT-16 | ops.rs:117,118 | u32 0 % 1, binary and compound-assignment spelling | zl=0, one=1 | uints[8,9] | 0 | exact | % u32 zero [new] | source |
| CT-17 | ops.rs:171,172 | u32 0 & 0, binary and compound-assignment spelling | zl=0, zr=0 | uints[10,11] | 0 | exact | & u32 zero [new] | source |
| CT-18 | ops.rs:173,174 | u32 0 \| 0, binary and compound-assignment spelling | zl=0, zr=0 | uints[12,13] | 0 | exact | \| u32 zero [new] | source |
| CT-19 | ops.rs:175,176 | u32 0 ^ 0, binary and compound-assignment spelling | zl=0, zr=0 | uints[14,15] | 0 | exact | ^ u32 zero [new] | source |
| CT-20 | ops.rs:235,236 | u32 0 << 0, binary and compound-assignment spelling | zl=0, sh=0 | uints[16,17] | 0 | exact | << u32 zero [new] | source |
| CT-21 | ops.rs:237,238 | u32 0 >> 0, binary and compound-assignment spelling | zl=0, sh=0 | uints[18,19] | 0 | exact | >> u32 zero [new] | source |
| CT-22 | ops.rs:54 | f32 -0.0 (unary minus), compared with == | zl=0.0 | floats[0] | 0.0 | exact | neg f32 zero [new] | source |
| CT-23 | ops.rs:109,110 | f32 0 + 0, binary and compound-assignment spelling | zl=0.0, zr=0.0 | floats[1,2] | 0.0 | exact | + f32 zero [new] | source |
| CT-24 | ops.rs:111,112 | f32 0 - 0, binary and compound-assignment spelling | zl=0.0, zr=0.0 | floats[3,4] | 0.0 | exact | - f32 zero [new] | source |
| CT-25 | ops.rs:113,114 | f32 0 * 0, binary and compound-assignment spelling | zl=0.0, zr=0.0 | floats[5,6] | 0.0 | exact | * f32 zero [new] | source |
| CT-26 | ops.rs:115,116 | f32 0 / 1, binary and compound-assignment spelling | zl=0.0, one=1.0 | floats[7,8] | 0.0 | GPU: 2.5 ULP set | / f32 zero [new] | source |
| CT-27 | ops.rs:56 | bool !true | t1=true | bools[0] | false | exact | ! bool true [existing] | source |
| CT-28 | ops.rs:171,172 (bool at 141 and 166) | bool false & false, binary and compound-assignment spelling | f1=false, f2=false | bools[1,2] | false | exact | & bool false [new] | source |
| CT-29 | ops.rs:173,174 (bool at 141 and 166) | bool false \| false, binary and compound-assignment spelling | f1=false, f2=false | bools[3,4] | false | exact | \| bool false [new] | source |
| CT-30 | ops.rs:175,176 (bool at 141 and 166) | bool false ^ false, binary and compound-assignment spelling | f1=false, f2=false | bools[5,6] | false | exact | ^ bool false [new] | source |
| CT-31 | int_macros.rs:20-22 via mod.rs:95-99 | i32 10 + 2 (the source asserts method == operator; value derived) | ten=10, two=2 | ints[21] | 12 | exact | + i32 small [existing] | derived |
| CT-32 | int_macros.rs:20-22 via mod.rs:95-99 | i32 10 - 2 (the source asserts method == operator; value derived) | ten=10, two=2 | ints[22] | 8 | exact | - i32 small [existing] | derived |
| CT-33 | int_macros.rs:20-22 via mod.rs:95-99 | i32 10 * 2 (the source asserts method == operator; value derived) | ten=10, two=2 | ints[23] | 20 | exact | * i32 small [existing] | derived |
| CT-34 | int_macros.rs:20-22 via mod.rs:95-99 | i32 10 / 2 (the source asserts method == operator; value derived) | ten=10, two=2 | ints[24] | 5 | exact | / i32 small [existing] | derived |
| CT-35 | int_macros.rs:20-22 via mod.rs:95-99 | i32 10 % 2 (the source asserts method == operator; value derived) | ten=10, two=2 | ints[25] | 0 | exact | % i32 small [existing] | derived |
| CT-36 | uint_macros.rs:15-18 via mod.rs:95-99 | u32 10 + 2 (the source asserts method == operator; value derived) | ten=10, two=2 | uints[20] | 12 | exact | + u32 small [existing] | derived |
| CT-37 | uint_macros.rs:15-18 via mod.rs:95-99 | u32 10 - 2 (the source asserts method == operator; value derived) | ten=10, two=2 | uints[21] | 8 | exact | - u32 small [existing] | derived |
| CT-38 | uint_macros.rs:15-18 via mod.rs:95-99 | u32 10 * 2 (the source asserts method == operator; value derived) | ten=10, two=2 | uints[22] | 20 | exact | * u32 small [existing] | derived |
| CT-39 | uint_macros.rs:15-18 via mod.rs:95-99 | u32 10 / 2 (the source asserts method == operator; value derived) | ten=10, two=2 | uints[23] | 5 | exact | / u32 small [existing] | derived |
| CT-40 | uint_macros.rs:15-18 via mod.rs:95-99 | u32 10 % 2 (the source asserts method == operator; value derived) | ten=10, two=2 | uints[24] | 0 | exact | % u32 small [existing] | derived |
| CT-41 | floats.rs:453-456 (test num) | f32 10.0 + 2.0 (the source asserts method == operator; value derived) | ten=10.0, two=2.0 | floats[9] | 12.0 | exact | + f32 small [existing] | derived |
| CT-42 | floats.rs:453-456 (test num) | f32 10.0 - 2.0 (the source asserts method == operator; value derived) | ten=10.0, two=2.0 | floats[10] | 8.0 | exact | - f32 small [existing] | derived |
| CT-43 | floats.rs:453-456 (test num) | f32 10.0 * 2.0 (the source asserts method == operator; value derived) | ten=10.0, two=2.0 | floats[11] | 20.0 | exact | * f32 small [existing] | derived |
| CT-44 | floats.rs:453-456 (test num) | f32 10.0 / 2.0 (the source asserts method == operator; value derived) | ten=10.0, two=2.0 | floats[12] | 5.0 | GPU: 2.5 ULP set | / f32 small [existing] | derived |
| CT-45 | i32.rs:6 | i32 a * (a - 1) | ten=10 | ints[26] | 90 | exact | * i32 small [existing]; - i32 small [existing] | source |
| CT-46 | i32.rs:9 | i32 a - 10 | ten=10 | ints[27] | 0 | exact | - i32 small [existing] | source |
| CT-47 | i32.rs:10 | i32 a / 10 | ten=10 | ints[28] | 1 | exact | / i32 small [existing] | source |
| CT-48 | i32.rs:11 | i32 a - 20 | ten=10 | ints[29] | -10 | exact | - i32 small [existing] | source |
| CT-49 | i32.rs:12 | i32 a << 10 | ten=10 | ints[30] | 10240 | exact | << i32 small [existing] | source |
| CT-50 | i32.rs:13 | i32 a << 16 | ten=10 | ints[31] | 655360 | exact | << i32 small [existing] | source |
| CT-51 | i32.rs:14 | i32 a * 16 | ten=10 | ints[32] | 160 | exact | * i32 small [existing] | source |
| CT-52 | i32.rs:15 | i32 a * a * a | ten=10 | ints[33] | 1000 | exact | * i32 small [existing] | source |
| CT-53 | i32.rs:16 | i32 a * a * a * a | ten=10 | ints[34] | 10000 | exact | * i32 small [existing] | source |
| CT-54 | i32.rs:17 | i32 a * a / a * a | ten=10 | ints[35] | 100 | exact | * i32 small [existing]; / i32 small [existing] | source |
| CT-55 | i32.rs:18 | i32 a * (a - 1) << (2 + a as usize), shift amount as u32 | ten=10 | ints[36] | 368640 | exact | * i32 small [existing]; - i32 small [existing]; << i32 small [existing]; + u32 small [existing]; as i32->u32 small [existing] | source |
| CT-56 | i32.rs:20 | i32 b + 1 - 1 (b = 0x10101010) | pat=0x10101010 | ints[37] | 269488144 | exact | + i32 pattern [new]; - i32 pattern [new] | source |
| CT-57 | i32.rs:23 | i32 b & b << 1 | pat=0x10101010 | ints[38] | 0 | exact | & i32 pattern [new]; << i32 pattern [new] | source |
| CT-58 | i32.rs:24 | i32 b \| b << 1 | pat=0x10101010 | ints[39] | 808464432 | exact | \| i32 pattern [new]; << i32 pattern [new] | source |
| CT-59 | int_macros.rs:14 | i32 MAX > 0 | hi=i32::MAX | ints[40] | true | exact | > i32 boundary [new] | source |
| CT-60 | int_macros.rs:15 | i32 MIN <= 0 | lo=i32::MIN | ints[41] | true | exact | <= i32 boundary [new] | source |
| CT-61 | int_macros.rs:16 | i32 MIN + MAX + 1 (= -1 + 1, no overflow) | lo=i32::MIN, hi=i32::MAX | ints[42] | 0 | exact | + i32 boundary [new] | source |
| CT-62 | int_macros.rs:26 | i32 0b1100 \| 0b1010 (source: .bitor) | m12=0b1100, m10=0b1010 | ints[43] | 14 | exact | \| i32 pattern [new] | source |
| CT-63 | int_macros.rs:27 | i32 0b1100 & 0b1010 (source: .bitand) | m12=0b1100, m10=0b1010 | ints[44] | 8 | exact | & i32 pattern [new] | source |
| CT-64 | int_macros.rs:28 | i32 0b1100 ^ 0b1010 (source: .bitxor) | m12=0b1100, m10=0b1010 | ints[45] | 6 | exact | ^ i32 pattern [new] | source |
| CT-65 | int_macros.rs:29 | i32 0b0111 << 1 (source: .shl) | m7=0b0111 | ints[46] | 14 | exact | << i32 pattern [new] | source |
| CT-66 | int_macros.rs:30 | i32 0b1110 >> 1 (source: .shr) | m14=0b1110 | ints[47] | 7 | exact | >> i32 pattern [new] | source |
| CT-67 | int_macros.rs:31 | i32 -(0b11) - 1 and !0b11 (source: .not), both sides | m3=0b11 | ints[48,49] | -4 | exact | ! i32 pattern [new]; neg i32 small [existing]; - i32 small [existing] | derived |
| CT-68 | uint_macros.rs:10 | u32 MAX > 0 | hi=u32::MAX | uints[25] | true | exact | > u32 boundary [new] | source |
| CT-69 | uint_macros.rs:11 | u32 MIN <= 0 | lo=0 (u32::MIN) | uints[26] | true | exact | <= u32 boundary [new] | source |
| CT-70 | uint_macros.rs:22 | u32 0b1100 \| 0b1010 (source: .bitor) | m12=0b1100, m10=0b1010 | uints[27] | 14 | exact | \| u32 pattern [new] | source |
| CT-71 | uint_macros.rs:23 | u32 0b1100 & 0b1010 (source: .bitand) | m12=0b1100, m10=0b1010 | uints[28] | 8 | exact | & u32 pattern [existing] | source |
| CT-72 | uint_macros.rs:24 | u32 0b1100 ^ 0b1010 (source: .bitxor) | m12=0b1100, m10=0b1010 | uints[29] | 6 | exact | ^ u32 pattern [new] | source |
| CT-73 | uint_macros.rs:25 | u32 0b0111 << 1 (source: .shl) | m7=0b0111 | uints[30] | 14 | exact | << u32 pattern [new] | source |
| CT-74 | uint_macros.rs:26 | u32 0b1110 >> 1 (source: .shr) | m14=0b1110 | uints[31] | 7 | exact | >> u32 pattern [new] | source |
| CT-75 | uint_macros.rs:27 | u32 MAX - 0b1011 and !0b1011 (source: .not), both sides | hi=u32::MAX, m11=0b1011 | uints[32,33] | 4294967284 | exact | - u32 boundary [new]; ! u32 pattern [existing] | derived |
| CT-76 | floats.rs:557-558 | f32 0.0 == -0.0 (the bit-equality at 559 is excluded) | nz=-0.0 | floats[13] | true | exact | == f32 signed-zero [new] | source |
| CT-77 | floats.rs:2066,2076 | i32 16777214 as f32, and (16777214 as f32) as i32 | x=16777214 | casts[0] | 16777214.0; 16777214 | exact | as i32->f32 boundary [new]; as f32->i32 boundary [new] | derived |
| CT-78 | floats.rs:2067,2077 | i32 16777215 as f32, and (16777215 as f32) as i32 | x=16777215 | casts[1] | 16777215.0; 16777215 | exact | as i32->f32 boundary [new]; as f32->i32 boundary [new] | derived |
| CT-79 | floats.rs:2068,2078 | i32 16777216 as f32, and (16777216 as f32) as i32 | x=16777216 | casts[2] | 16777216.0; 16777216 | exact | as i32->f32 boundary [new]; as f32->i32 boundary [new] | derived |
| CT-80 | floats.rs:2071-2073,2081,2085-2087 | i32 16777217 as f32, and (16777217 as f32) as i32 | x=16777217 | casts[3] | 16777216.0; 16777216 | GPU: two-neighbour set | as i32->f32 boundary [new]; as f32->i32 boundary [new] | derived |
| CT-81 | floats.rs:2110,2120 | i32 -16777214 as f32, and (-16777214 as f32) as i32 | x=-16777214 | casts[4] | -16777214.0; -16777214 | exact | as i32->f32 boundary [new]; as f32->i32 boundary [new] | derived |
| CT-82 | floats.rs:2111,2121 | i32 -16777215 as f32, and (-16777215 as f32) as i32 | x=-16777215 | casts[5] | -16777215.0; -16777215 | exact | as i32->f32 boundary [new]; as f32->i32 boundary [new] | derived |
| CT-83 | floats.rs:2112,2122 | i32 -16777216 as f32, and (-16777216 as f32) as i32 | x=-16777216 | casts[6] | -16777216.0; -16777216 | exact | as i32->f32 boundary [new]; as f32->i32 boundary [new] | derived |
| CT-84 | floats.rs:2115-2117,2125,2129-2131 | i32 -16777217 as f32, and (-16777217 as f32) as i32 | x=-16777217 | casts[7] | -16777216.0; -16777216 | GPU: two-neighbour set | as i32->f32 boundary [new]; as f32->i32 boundary [new] | derived |

Known gap found by CT-30 (`bool ^`, in the `bools` kernel with CT-27 to CT-29): the validator accepts `^` on `bool`,
but the emitter passes `^` and `^=` through, so `slangc` writes `var _S4 : bool = (f1_0 ^ (f2_0));` and naga 30.0.1
rejects it (section 6: WGSL defines `^` on integers only). The case stays in the table and the kernel on purpose.
The fix is a separate slice that lowers `^` and `^=` on `bool` through a gated `__gust_xor` overload, like
`__gust_not`; it must land before slice m5.d2, because until then the whole `bools` kernel, so CT-27 to CT-30, fails
at shader creation on a GPU. The twin is unaffected: it runs Rust's `^`.

Resolved: the fix (slice fx.bxor, `__gust_xor`) has merged, and CT-30 passes on both adapters of section 7.

## 4. Unique coverage

The record counts unique coverage, not assertion volume (R1-7). A triple is (operator, type, operand class); a row
can exercise several. A triple is `existing` when a kernel of the repository already executes the same
operator on the same type with an operand of the same class inside the domain: a pinned `DOMAIN` control of
`crates/gust-wgpu/tests/mismatch.rs` (GPU, twin and plain Rust) or an operation in another file of `tests/fixtures/`
(the evidence column names the case). The class is a judgement; the evidence is the column. An operator-type pair is
`existing` when any of those kernels runs that operator on that type at all. The kernels of `examples/` and of the
other `#[gpu]` modules in `crates/*/tests/` were also searched, by reading their operator lines, for the pairs reported
`new`, and none uses them.

Unique coverage: triples=70 new=50 existing=20

Unique operator-type pairs: pairs=40 new=13 existing=27

Operator-type pairs that no fixture runs at all: `! i32`, `& bool`, `& i32`, `<= i32`, `<= u32`, `> i32`, `^ bool`, `^ i32`, `^ u32`, `as i32->f32`, `| bool`, `| i32`, `| u32`.

Evidence for the pairs already run: `+ - * / % << >>` and `neg` on `i32`, and the same without `neg` on `u32`, are the
`DOMAIN` controls of the `idivide`, `ishift`, `iarith`, `udivide`, `ushift` and `uarith` kernels of `mismatch.rs`;
`& u32` and `! u32` are in `semantics.rs` and `option.rs`; `> u32` is `loops.rs:87` and `option.rs:50`; `+ - * / == neg`
on `f32` are `float_op`, `float_cmp` (`mismatch.rs`) and `semantics.rs`; `! bool` is `semantics.rs`; `as f32->i32` is
`cast_int` (`mismatch.rs`) and `numeric.rs`; `as i32->u32` is `loops.rs:94`.

The transcription has 84 rows asserting 121 output slots. The M5 plan pre-registered an estimate of 20 to 40
assertions as an estimate and not as a target; the row count is higher because the 22 operator-defined tests of
`ops.rs` give 30 operation rows over `i32`, `u32`, `f32` and `bool`, and the exact-integer tests give eight conversion
rows. These are not padding to a target: the unique coverage above is the number that counts, and it shows where the
volume repeats (the `ops.rs` rows are all `zero`-class operands). 24 rows have a derived literal; 2 rows carry
the 2.5 ULP set and 2 rows the two-neighbour set.

The unique triples follow, with the evidence for each `existing` one (judged by reading the fixtures at the commit
this page was written against). A `new` triple is exercised by no fixture.
The listing below is informative: only the Triples column of the table is bound by the test, so this listing can drift
from that column without a test failing.

| Triple | Existing coverage |
| --- | --- |
| ! bool true | semantics.rs `!(i < 2u32)` |
| ! u32 pattern | semantics.rs `!x & 0xff_u32` |
| % i32 small | mismatch.rs DOMAIN control `i32 -7 % 2` |
| % u32 small | mismatch.rs DOMAIN control `u32 7 % 2` |
| & u32 pattern | semantics.rs `x & 1u32` |
| * f32 small | mismatch.rs DOMAIN control `f32 2.0 * 3.0` |
| * i32 small | mismatch.rs DOMAIN control `i32 -6 * 7` |
| * u32 small | mismatch.rs DOMAIN control `u32 6 * 7` |
| + f32 small | mismatch.rs DOMAIN control `f32 1.5 + 2.25` |
| + i32 small | mismatch.rs DOMAIN control `i32 -5 + 3` |
| + u32 small | mismatch.rs DOMAIN control `u32 2 + 3` |
| - f32 small | mismatch.rs DOMAIN control `f32 1.0 - 2.0` |
| - i32 small | mismatch.rs DOMAIN control `i32 3 - 5` |
| - u32 small | mismatch.rs DOMAIN control `u32 5 - 3` |
| / f32 small | mismatch.rs DOMAIN control `f32 1.0 / 4.0` |
| / i32 small | mismatch.rs DOMAIN control `i32 -7 / 2` |
| / u32 small | mismatch.rs DOMAIN control `u32 7 / 2` |
| << i32 small | mismatch.rs DOMAIN control `i32 1 << 30` |
| as i32->u32 small | loops.rs `(neg_sum + 10i32) as uint` |
| neg i32 small | mismatch.rs DOMAIN control `i32 -(5)` |
| ! i32 pattern | new |
| % i32 zero | new |
| % u32 zero | new |
| & bool false | new |
| & i32 pattern | new |
| & i32 zero | new |
| & u32 zero | new |
| * f32 zero | new |
| * i32 zero | new |
| * u32 zero | new |
| + f32 zero | new |
| + i32 boundary | new |
| + i32 pattern | new |
| + i32 zero | new |
| + u32 zero | new |
| - f32 zero | new |
| - i32 pattern | new |
| - i32 zero | new |
| - u32 boundary | new |
| - u32 zero | new |
| / f32 zero | new |
| / i32 zero | new |
| / u32 zero | new |
| << i32 pattern | new |
| << i32 zero | new |
| << u32 pattern | new |
| << u32 zero | new |
| <= i32 boundary | new |
| <= u32 boundary | new |
| == f32 signed-zero | new |
| > i32 boundary | new |
| > u32 boundary | new |
| >> i32 pattern | new |
| >> i32 zero | new |
| >> u32 pattern | new |
| >> u32 zero | new |
| ^ bool false | new |
| ^ i32 pattern | new |
| ^ i32 zero | new |
| ^ u32 pattern | new |
| ^ u32 zero | new |
| as f32->i32 boundary | new |
| as i32->f32 boundary | new |
| neg f32 zero | new |
| neg i32 zero | new |
| \| bool false | new |
| \| i32 pattern | new |
| \| i32 zero | new |
| \| u32 pattern | new |
| \| u32 zero | new |

## 5. Exclusions

Each exclusion has its reason. "Dialect" means `docs/SEMANTICS.md` section 1, "Language subset", and the README
"Current subset".

- `ops.rs`, every width other than 32-bit (`i8`, `i16`, `i64`, `i128`, `isize`, `u8`, `u16`, `u64`, `u128`, `usize`,
  `f64`) and the shift right-hand-type matrix (lines 178-233): types outside the dialect. The matrix varies only the
  type of the zero shift amount; the rows use `u32`, the shift-amount type of the existing fixtures.
- `ops.rs` float `%` (lines 117, 118) and `floats.rs` `num_rem` (462-474): no bound is stated for float `%`
  (`docs/SEMANTICS.md` section 4; R1-7). Float `%` enters with a later version of the difftest bound.
- `ops.rs` reference forms (`L op &R`, `&L op R`, `&L op &R`, `&mut L op &R`): the dialect has no references.
- `mod.rs` `test_num` (84-100), the method half (`ten.add(two)` and the other trait-method spellings): method calls of
  operator traits are not in the dialect; the operator half is transcribed (derived literals).
- `mod.rs` lines 111-822 (parse, `from_str_radix`, `try_from`, `From` and `test_f32f64`): library functions and
  types outside the dialect (`FromStr`, `TryFrom`, `Result`, `u8` to `u128`, `f64`).
- `floats.rs`, the method tests: `min`, `max`, `minimum`, `maximum`, `midpoint`, `midpoint_large_magnitude`, `abs`,
  `copysign`, `rem_euclid`, `div_euclid`, `floor`, `ceil`, `round`, `round_ties_even`, `trunc`, `fract`, `signum`,
  `is_*`, `classify`, `next_up`, `next_down`, `sqrt_domain`, `clamp*`, `total_cmp*`, `recip`, `powi`, `powf`, `exp*`,
  `ln*`, `log*`, `asinh`, `acosh`, `atanh`, `gamma`, `ln_gamma`, `to_degrees`, `to_radians`, `to_algebraic`,
  `to_bits_conv`, `mul_add`: method calls, and the validator accepts no method other than `.len()`, `.global_id()` and
  the `Option` and `Result` helpers. Several also use associated constants (`Float::MAX`) or `f64`.
- `floats.rs` `nan`, `infinity`, `neg_infinity` and every NaN or infinity assertion inside other tests: outside the
  defined domain (`docs/SEMANTICS.md` section 3, item 5; D39).
- `floats.rs` `neg_zero`, the bit-equality `assert_biteq!(-0.0, neg_zero)` (line 559) and the sign checks: a GPU may
  ignore the sign of a zero (15.7.2). The comparison at line 558 is transcribed.
- `floats.rs` `zero`, `one`, `real_consts`: associated constants and `f64`/`f16`/`f128` content, no operation.
- `floats.rs` `from` (1992-2010), `from_u16_i16` (2012-2027), `from_u32_i32` (2029-2046): `Float::from` of `bool`, `u8`,
  `i8`, `u16`, `i16`: types outside the dialect (`from_u32_i32` is `cfg(false)` for `f32`).
- `int_macros.rs` and `uint_macros.rs`, every test except `test_overflows` (comparisons and the sum), `test_num` and
  `test_bitwise_operators`: method tests (`rem_euclid`, `abs`, `signum`, `is_positive`, `count_ones`, `leading_ones`,
  `bit_width`, `rotate_*`, `funnel_*`, `carryless_mul`, `swap_bytes`, `extract_bits`, `deposit_bits`, `reverse_bits`,
  `from_le`, `from_be`, `checked_*`, `saturating_*`, `isolate_*`, `highest_one`, `lowest_one`, `from_str*`, `pow`,
  `isqrt`, `div_floor`, `div_ceil`, `next_multiple_of`, `carrying_*`, `borrowing_sub`, `midpoint`, `unbounded_*`,
  `div_exact`).
- `uint_macros.rs:12`, `(MIN + MAX).wrapping_add(1) == 0`: left out when the corpus was transcribed, because
  `wrapping_add` was a method then and a plain `MAX + 1` wraps, which is outside the domain (KM-10). Since slice m7.w
  (D87) `wrapping_add` on `u32` and `i32` is in the dialect and inside the domain, so this case could now be
  transcribed.
- `i32.rs:7-8`: `assert_eq!(a, 10)` compares a variable with its own literal; there is no operation.
- `i32.rs:21, 22, 26-29`: both sides are the same expression, so the source states no expected value to transcribe.
- `i32.rs`, `isize` operands: transcribed to `i32`, every value fits (the largest is 368640 and 0x30303030).
- `cast.rs`: `checked_cast`, `wrapping_cast`, `saturating_cast` on every integer width: methods and widths outside the
  dialect.
- `midpoint.rs`, `int_log.rs`, `int_sqrt.rs`, `clamp_magnitude.rs`, `carryless_mul.rs`: method tests.
  `wrapping.rs`: `std::num::Wrapping<T>`, a library type. `const_from.rs`: `From` and `TryFrom` in `const` contexts.
  `nan.rs`: formatting of `f64` NaN. `float_iter_sum_identity.rs`: iterator sums and `1.0 / -0.0`.
- Not read, excluded by subject (listed by the repository's directory listing, never opened): `bignum.rs`,
  `float_ieee754_flt2dec_dec2flt.rs` and the directories `dec2flt/` and `flt2dec/` (number formatting and parsing,
  big-number arithmetic), `niche_types.rs`, and the one-line files `i8.rs`, `i16.rs`, `i64.rs`, `i128.rs`, `u8.rs`,
  `u16.rs`, `u64.rs`, `u128.rs` (a macro call for a type outside the dialect).
- Float `+`, `-`, `*` whose exact result is not representable: none occurs in the selected cases, so none is excluded
  for that reason (R1-7). The float rows are `0 op 0`, `10.0 op 2.0` and the conversions.
- Integers above 2^24 other than 2^24 + 1 and below -2^24 other than -(2^24 + 1) in the conversion rows: the source
  stops at `MAX_EXACT_INTEGER + 2` and its mirror, and the rows do not add values the source does not test.

## 6. What the twin test checks, and what is not claimed

`crates/gust/tests/coretests_twin.rs` has five tests:

- the twin of every kernel equals every literal of the table, in all 121 output slots;
- every `CT-nn` of this table has a case and every case is in the table, ids run `CT-01` to `CT-84` without a
  gap, and every id appears in the fixture;
- the kernel slots, the expected literals and the acceptance sets of this table equal the cases;
- every output slot of every kernel is asserted by exactly one case;
- the unique-coverage numbers above equal what the Triples column says.

Not claimed by the twin test: GPU behaviour of any row (section 7 records it), the labels of `docs/SEMANTICS.md`
section 2 (nothing here is `measured` or `proven`), or equivalence of the twin and the GPU.

Observations from writing the fixture, all about this repository and not about the Rust sources:

- The five kernels translate and compile to WGSL with `slangc` (a throwaway test, not part of the repository; SPIR-V
  was not checked).
- wgpu loads WGSL through naga, so the five WGSL outputs were also parsed and validated with naga 30.0.1 (the locked
  version; `wgsl-in`, `ValidationFlags::all()`, `Capabilities::empty()`) by a throwaway probe crate outside the
  repository (`plans/m5/probe/naga-check/` of the run directory, offline build). `ints`, `uints`, `floats` and `casts`
  parse and validate. `bools` parses but fails validation: `Operation ExclusiveOr can't work with` two `Bool`
  operands, at `var _S4 : bool = (f1_0 ^ (f2_0));`, which is CT-30 (see "Known gap" below the table). `bools` is
  expected to fail until the `__gust_xor` slice lands. With that one expression replaced by `!=` in a copy of the WGSL,
  `bools` validates, so CT-30 is the only defect found.
- In the WGSL of the `bools` kernel `slangc` merges the binary and the compound-assignment spelling of one operation
  into one expression, so on a GPU the second spelling of such a row does not execute a separate instruction.
- Slice m5.d2 must use the input arrays and the output lengths of `crates/gust/tests/coretests_twin.rs` (the
  constants `INTS_A` to `BOOLS_OUT`, lines 30 to 69) unchanged, element for element, so that the slots and variables of
  the table mean the same thing on the GPU.

## 7. GPU results (slice m5.d2)

`crates/gust-wgpu/tests/coretests.rs` (Gpu tier) includes the fixture, dispatches the five kernels on the adapter with
the input arrays and output lengths of the twin test unchanged, runs the twin on the same inputs, and compares every
output slot with the literal of the table: the adapter under the acceptance set of the row, the twin exact. It prints
the adapter (name, backend, device type, driver, `driver_info`) and, for the two-neighbour and `/` rows, the observed
value. A failure names the case id, the accepted set, and the GPU and twin values. The id guard records each case id
as it runs and requires the executed set to equal the ids of the table above (each exactly once, none unknown); a second
check requires the slot, literal and acceptance columns of every table row to equal what the test's own cases assert,
rejects a duplicate row, and requires every output slot of every kernel to be asserted by exactly one case (the
twin test's guards, ported; the two files are not tied by `include_str!`, so the table is their common reference).
Nothing is skipped, ignored or loosened. The run was a debug-profile build (`GUST_PROFILE=dev`) on 2026-11-01.

Adapters:

- **RTX 5090**: `NVIDIA GeForce RTX 5090`, backend Vulkan, device type `DiscreteGpu`, driver `NVIDIA`, `driver_info`
  `610.88` (Windows).
- **lavapipe**: `llvmpipe (LLVM 20.1.2, 256 bits)`, backend Vulkan, device type `Cpu`, driver `llvmpipe`, `driver_info`
  `Mesa 25.2.8-0ubuntu0.24.04.4 (LLVM 20.1.2)` (WSL Ubuntu 24.04, `VK_DRIVER_FILES` set to the lavapipe ICD).

Per group of rows (the label applies to every row of the group; "pass" means all slots of all rows were inside the
acceptance set and the twin equalled the literal):

| Group | Rows | Count | Acceptance | RTX 5090 | lavapipe |
| --- | --- | --- | --- | --- | --- |
| i32 operators, comparisons, `-`, `!` | CT-01 to CT-11, CT-31 to CT-35, CT-45 to CT-67 | 39 | exact | pass | pass |
| u32 operators, comparisons, `!` | CT-12 to CT-21, CT-36 to CT-40, CT-68 to CT-75 | 23 | exact | pass | pass |
| f32 `+ - *`, unary minus, `==` | CT-22 to CT-25, CT-41 to CT-43, CT-76 | 8 | exact | pass | pass |
| f32 `/` | CT-26, CT-44 | 2 | GPU: 2.5 ULP set | pass, 0 ULP | pass, 0 ULP |
| bool `! & \| ^` | CT-27 to CT-30 | 4 | exact | pass | pass |
| i32 to f32 to i32 at and below 2^24 | CT-77 to CT-79, CT-81 to CT-83 | 6 | exact | pass | pass |
| i32 to f32 to i32 above 2^24 | CT-80, CT-84 | 2 | GPU: two-neighbour set | pass, nearest even | pass, nearest even |

Both adapters executed 84 cases and 121 output slots, and no case failed on either.

What the runs show and what they do not:

- The float `/` rows are weak evidence for the 2.5 ULP set: `0.0 / 1.0` and `10.0 / 2.0` are exactly representable and
  both adapters returned them exactly (0 ULP). With the set tightened to exact, no case fails on either adapter, so the
  rows do not exercise the slack that WGSL allows; they are kept because the set is the contract (R1-7), not because
  the slack was observed. The 2.5 ULP branch (and the 16777218 neighbour branch) is covered by a host-only unit test of
  the acceptance function in `coretests.rs` with synthetic values, not by data: the literal and 1 and 2 ULP either way
  pass, 3 ULP, NaN and infinity fail, zeros of either sign agree, and both neighbours pass while the source integer
  fails. No non-exact `/` row was added, because the table is a transcription. R1-7's `ulps <= 2` is stricter than
  WGSL's 2.5 ULP just below a power of two, where the spacing halves; that matters only for future rows.
- Both adapters converted 16777217 and -16777217 to the nearest even neighbour (16777216 and -16777216), which is the
  twin's value; the set also accepts 16777218 and -16777218, and neither adapter used that slack. This is an
  observation of two adapters, not a bound, and it changes no label of `docs/SEMANTICS.md` section 2.
- CT-30 (`bool ^`) runs on both adapters, which is the regression for the `__gust_xor` lowering.
- Nothing here changes a statement of `docs/SEMANTICS.md`; the values are inside the defined domain, so a difference on
  any adapter would have been a failure to investigate, not a second observation.

## 8. Attribution and licence

The cases are transcribed from the Rust core tests, copyright The Rust Project Developers, licensed MIT OR Apache-2.0,
as is this repository. This page and the fixture take operations, operand values and expected results only. No test
function, comment or prose of the Rust sources is copied; the descriptions in the table are our own.
