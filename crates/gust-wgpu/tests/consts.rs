//! A module-level `const` shared by the kernels of a `#[gpu]` module, on the GPU and on the CPU twin (slice m11.g3,
//! D205; Gpu tier).
//!
//! The module below holds four `pub const` items (a `u32`, a negative `i32`, an `f32` and a `bool`, plus a `u32` that is
//! the largest value) read by a helper and by two kernels. The macro lowers each to one `static const` line in the text
//! of every kernel and leaves the Rust item as it is, so the host reads the same items: the expected values below are
//! computed from `shared::WIDTH` and the others, never from a copy of the number. Every value is a small integer or a
//! binary fraction, so each result is exactly representable and the comparisons are exact.
//!
//! The twin is a second assertion and never GPU evidence.
//!
//! Slice m11.cexpr (D210) adds the module `cexpr`: constants that are expressions of literals and earlier constants,
//! read back by one kernel and compared with the values the host reads from the same Rust items (rustc computed those).

mod common;

use gust::{Independent, cpu::CpuDevice};
use gust_wgpu::{BufferBinding, GpuBufferAccess};

#[gust::gpu]
mod shared {
    pub const WIDTH: u32 = 4096;
    pub const OFFSET: i32 = -7;
    pub const SCALE: f32 = 0.25;
    pub const ON: bool = true;
    pub const OFF: bool = false;
    pub const MASK: u32 = 4294967295;
    pub const LOW: i32 = -2147483648;
    pub const LOW_SUFFIXED: i32 = -2147483648i32;
    pub const NEG_REAL: f32 = -1.5;
    pub const TWO: f32 = 2f32;

    pub struct Acc {
        pub n: u32,
    }

    impl Acc {
        // A method of a struct reads a module const.
        pub fn bumped(&self) -> u32 {
            self.n + WIDTH
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn extras(
        id: SV_DispatchThreadID,
        src: StructuredBuffer<u32>,
        mut lows: RWStructuredBuffer<i32>,
        mut lows_suffixed: RWStructuredBuffer<i32>,
        mut negs: RWStructuredBuffer<f32>,
        mut methods: RWStructuredBuffer<u32>,
        mut closures: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        if i < src.len() {
            lows[i] = LOW + (src[i] as i32);
            lows_suffixed[i] = LOW_SUFFIXED + (src[i] as i32);
            negs[i] = NEG_REAL * TWO * (src[i] as f32);
            let acc = Acc { n: src[i] };
            methods[i] = acc.bumped();
            // A closure reads a module const without capturing it.
            let add = |x: u32| -> u32 { x + WIDTH };
            let once = add(src[i]);
            closures[i] = add(once - WIDTH);
        }
    }

    fn lift(x: u32) -> u32 {
        if ON { x + WIDTH } else { x }
    }

    fn shift(x: u32) -> i32 {
        (x as i32) + OFFSET
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn integers(
        id: SV_DispatchThreadID,
        src: StructuredBuffer<u32>,
        mut wide: RWStructuredBuffer<u32>,
        mut signed: RWStructuredBuffer<i32>,
    ) {
        let i = id.x;
        if i < src.len() {
            wide[i] = lift(src[i]) + WIDTH;
            signed[i] = shift(src[i]) + OFFSET;
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn reals(
        id: SV_DispatchThreadID,
        src: StructuredBuffer<u32>,
        mut scaled: RWStructuredBuffer<f32>,
        mut flags: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        if i < src.len() {
            scaled[i] = (lift(src[i]) as f32) * SCALE;
            let mut on: u32 = 0u32;
            if ON {
                on = 1u32;
            }
            let mut off: u32 = 0u32;
            if OFF {
                off = 1u32;
            }
            flags[i] = ((src[i] ^ MASK) & 255u32) + on * 10u32 + off;
        }
    }
}

/// The element count of every dispatch: not a multiple of the workgroup width (64), so the guard matters.
const N: usize = 70;

fn sources() -> Vec<u32> {
    (0..N as u32).map(|k| k * 3).collect()
}

/// The host's expectation, computed from the same `pub const` items the shader reads.
struct Expected {
    wide: Vec<u32>,
    signed: Vec<i32>,
    scaled: Vec<f32>,
    flags: Vec<u32>,
}

fn expected(src: &[u32]) -> Expected {
    let on = u32::from(shared::ON);
    let off = u32::from(shared::OFF);
    Expected {
        wide: src
            .iter()
            .map(|&x| x + shared::WIDTH + shared::WIDTH)
            .collect(),
        signed: src
            .iter()
            .map(|&x| x as i32 + shared::OFFSET + shared::OFFSET)
            .collect(),
        scaled: src
            .iter()
            .map(|&x| (x + shared::WIDTH) as f32 * shared::SCALE)
            .collect(),
        flags: src
            .iter()
            .map(|&x| ((x ^ shared::MASK) & 255) + on * 10 + off)
            .collect(),
    }
}

#[test]
fn the_host_expectation_reads_the_constants_it_claims_to() {
    // The constants are what the test is about: a change to one must change the expectation, not leave it behind.
    assert_eq!(shared::WIDTH, 4096);
    assert_eq!(shared::OFFSET, -7);
    assert_eq!(shared::SCALE, 0.25);
    assert_eq!((shared::ON, shared::OFF), (true, false));
    assert_eq!(shared::MASK, u32::MAX);
    let e = expected(&[0, 3]);
    assert_eq!(e.wide, [8192, 8195]);
    assert_eq!(e.signed, [-14, -11]);
    assert_eq!(e.scaled, [1024.0, 1024.75]);
    assert_eq!(e.flags, [255 + 10, 252 + 10]);
}

#[test]
fn two_kernels_and_a_helper_read_the_same_module_constants_on_the_gpu_and_the_twin() {
    let src = sources();
    let want = expected(&src);

    let device = common::device();
    let info = device.adapter_info();
    eprintln!("adapter: {} backend: {:?}", info.name, info.backend);

    let source = device.create_typed_buffer("consts src", &src, GpuBufferAccess::ReadOnly);
    let wide = device.create_typed_buffer("consts wide", &[0u32; N], GpuBufferAccess::ReadWrite);
    let signed =
        device.create_typed_buffer("consts signed", &[0i32; N], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &shared::integers::DESCRIPTOR,
            N as u32,
            &[
                BufferBinding::read_only(&source).independent_length(),
                BufferBinding::read_write(&wide).independent_length(),
                BufferBinding::read_write(&signed).independent_length(),
            ],
        )
        .unwrap();
    let gpu_wide = device.read_typed_buffer(&wide).unwrap();
    let gpu_signed = device.read_typed_buffer(&signed).unwrap();
    assert_eq!(gpu_wide, want.wide, "gpu: integers, u32 constants");
    assert_eq!(gpu_signed, want.signed, "gpu: integers, i32 constants");

    let scaled =
        device.create_typed_buffer("consts scaled", &[0f32; N], GpuBufferAccess::ReadWrite);
    let flags = device.create_typed_buffer("consts flags", &[0u32; N], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &shared::reals::DESCRIPTOR,
            N as u32,
            &[
                BufferBinding::read_only(&source).independent_length(),
                BufferBinding::read_write(&scaled).independent_length(),
                BufferBinding::read_write(&flags).independent_length(),
            ],
        )
        .unwrap();
    let gpu_scaled = device.read_typed_buffer(&scaled).unwrap();
    let gpu_flags = device.read_typed_buffer(&flags).unwrap();
    assert_eq!(gpu_scaled, want.scaled, "gpu: reals, f32 constant");
    assert_eq!(gpu_flags, want.flags, "gpu: reals, bool and mask constants");
    eprintln!(
        "consts: elements={N} u32_exact=true i32_exact=true f32_exact=true bool_exact=true mask_exact=true"
    );

    // The twin: the same typed calls through `CpuDevice`, asserted against the host as well.
    let cpu = CpuDevice::new();
    let twin_src = cpu.buffer(&src);
    let twin_wide = cpu.buffer(&[0u32; N]);
    let twin_signed = cpu.buffer(&[0i32; N]);
    shared::integers::dispatch(
        &cpu,
        N as u32,
        Independent(&twin_src),
        Independent(&twin_wide),
        Independent(&twin_signed),
    )
    .unwrap();
    assert_eq!(twin_wide.read(), want.wide, "twin: integers, u32 constants");
    assert_eq!(
        twin_signed.read(),
        want.signed,
        "twin: integers, i32 constants"
    );
    let twin_scaled = cpu.buffer(&[0f32; N]);
    let twin_flags = cpu.buffer(&[0u32; N]);
    shared::reals::dispatch(
        &cpu,
        N as u32,
        Independent(&twin_src),
        Independent(&twin_scaled),
        Independent(&twin_flags),
    )
    .unwrap();
    assert_eq!(twin_scaled.read(), want.scaled, "twin: reals, f32 constant");
    assert_eq!(
        twin_flags.read(),
        want.flags,
        "twin: reals, bool and mask constants"
    );
}

#[test]
fn extreme_literals_a_struct_method_and_a_closure_read_constants_on_the_gpu_and_the_twin() {
    let src = sources();
    let lows: Vec<i32> = src.iter().map(|&x| shared::LOW + x as i32).collect();
    let lows_suffixed: Vec<i32> = src
        .iter()
        .map(|&x| shared::LOW_SUFFIXED + x as i32)
        .collect();
    let reals: Vec<f32> = src
        .iter()
        .map(|&x| shared::NEG_REAL * shared::TWO * x as f32)
        .collect();
    let methods: Vec<u32> = src.iter().map(|&x| x + shared::WIDTH).collect();
    let closures = methods.clone();
    assert_eq!(shared::LOW, i32::MIN);
    assert_eq!(shared::LOW_SUFFIXED, i32::MIN);

    let device = common::device();
    let info = device.adapter_info();
    eprintln!("adapter: {} backend: {:?}", info.name, info.backend);
    let source = device.create_typed_buffer("extras src", &src, GpuBufferAccess::ReadOnly);
    let g_lows = device.create_typed_buffer("extras lows", &[0i32; N], GpuBufferAccess::ReadWrite);
    let g_lows2 =
        device.create_typed_buffer("extras lows2", &[0i32; N], GpuBufferAccess::ReadWrite);
    let g_reals =
        device.create_typed_buffer("extras reals", &[0f32; N], GpuBufferAccess::ReadWrite);
    let g_methods =
        device.create_typed_buffer("extras methods", &[0u32; N], GpuBufferAccess::ReadWrite);
    let g_closures =
        device.create_typed_buffer("extras closures", &[0u32; N], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &shared::extras::DESCRIPTOR,
            N as u32,
            &[
                BufferBinding::read_only(&source).independent_length(),
                BufferBinding::read_write(&g_lows).independent_length(),
                BufferBinding::read_write(&g_lows2).independent_length(),
                BufferBinding::read_write(&g_reals).independent_length(),
                BufferBinding::read_write(&g_methods).independent_length(),
                BufferBinding::read_write(&g_closures).independent_length(),
            ],
        )
        .unwrap();
    assert_eq!(
        device.read_typed_buffer(&g_lows).unwrap(),
        lows,
        "gpu: i32::MIN, unsuffixed"
    );
    assert_eq!(
        device.read_typed_buffer(&g_lows2).unwrap(),
        lows_suffixed,
        "gpu: i32::MIN, suffixed"
    );
    assert_eq!(
        device.read_typed_buffer(&g_reals).unwrap(),
        reals,
        "gpu: negative and `2f32` f32 constants"
    );
    assert_eq!(
        device.read_typed_buffer(&g_methods).unwrap(),
        methods,
        "gpu: a struct method reads a constant"
    );
    assert_eq!(
        device.read_typed_buffer(&g_closures).unwrap(),
        closures,
        "gpu: a closure reads a constant"
    );

    let cpu = CpuDevice::new();
    let twin_src = cpu.buffer(&src);
    let t_lows = cpu.buffer(&[0i32; N]);
    let t_lows2 = cpu.buffer(&[0i32; N]);
    let t_reals = cpu.buffer(&[0f32; N]);
    let t_methods = cpu.buffer(&[0u32; N]);
    let t_closures = cpu.buffer(&[0u32; N]);
    shared::extras::dispatch(
        &cpu,
        N as u32,
        Independent(&twin_src),
        Independent(&t_lows),
        Independent(&t_lows2),
        Independent(&t_reals),
        Independent(&t_methods),
        Independent(&t_closures),
    )
    .unwrap();
    assert_eq!(t_lows.read(), lows, "twin: i32::MIN, unsuffixed");
    assert_eq!(t_lows2.read(), lows_suffixed, "twin: i32::MIN, suffixed");
    assert_eq!(
        t_reals.read(),
        reals,
        "twin: negative and `2f32` f32 constants"
    );
    assert_eq!(
        t_methods.read(),
        methods,
        "twin: a struct method reads a constant"
    );
    assert_eq!(
        t_closures.read(),
        closures,
        "twin: a closure reads a constant"
    );
}

// Slice m11.cexpr (D210): constants built from expressions of literals and earlier constants. The macro evaluates
// nothing: each Slang `static const` holds the expression the Rust item holds, operator for operator, so rustc computes
// the host's value and slangc computes the shader's, and the test compares the two. rustc already refuses an item that
// overflows or divides by zero.
#[gust::gpu]
mod cexpr {
    // `u32`: a product, a chain through three constants, results above `i32::MAX`, truncating `/` and `%`, shifts by a
    // constant, and `!`.
    pub const SIDE: u32 = 64;
    pub const CELLS: u32 = SIDE * SIDE;
    pub const STEP1: u32 = 5;
    pub const STEP2: u32 = STEP1 * 3 + 1;
    pub const STEP3: u32 = STEP2 * STEP2 - STEP1;
    pub const BASE: u32 = 3000000000;
    pub const HIGH: u32 = BASE + 1000000000;
    pub const PRODUCT: u32 = 65536 * 65535;
    pub const QUOT: u32 = HIGH / 7;
    pub const REM: u32 = HIGH % 1000;
    pub const SH: u32 = 5;
    pub const SHL: u32 = 1 << SH;
    pub const SHL_TOP: u32 = 1 << 31;
    pub const HALF_TOP: u32 = (1 << 31) / 2;
    pub const SHR: u32 = HIGH >> SH;
    pub const NOT: u32 = !SIDE;
    pub const BITS: u32 = (0xF0F0_F0F0 & 0x0FF0_0FF0) | (1 ^ 2);
    pub const MIXED: u32 = (HIGH - BASE) * 3 / 2 + SIDE % 5;
    // A bitwise not above `i32::MAX` fed to division, remainder and a right shift: unsigned operations on a value
    // that a signed one would read as negative.
    pub const NOT_HALF: u32 = !SIDE / 2;
    pub const NOT_REM: u32 = !SIDE % 7;
    pub const NOT_SHR: u32 = !SIDE >> 1;

    // `i32`: a negative result, `i32::MIN` by an expression, truncating `/` and `%` with a negative operand, an
    // arithmetic shift, a shift by a `u32` constant, and `!`.
    pub const NEG_PROD: i32 = -7 * 3;
    pub const I_MIN: i32 = -2147483647 - 1;
    pub const TRUNC_DIV: i32 = -7 / 2;
    pub const TRUNC_REM: i32 = -7 % 2;
    pub const DIV_NEG: i32 = 7 / -2;
    pub const REM_NEG: i32 = 7 % -2;
    pub const ASH: i32 = -16 >> 2;
    pub const ISHL: i32 = -3 << SH;
    pub const INOT: i32 = !5;
    pub const INOT_NEG: i32 = !NEG_PROD;
    pub const SPAN: i32 = NEG_PROD - I_MIN;
    // A shift into the sign bit, then a signed division and an arithmetic shift of the negative result.
    pub const SHL_DIV: i32 = (1 << 31) / 2;
    pub const SHL_SHR: i32 = (3 << 30) >> 1;

    // `f32`: results that are not representable, so a rounding happens, compared bit for bit with the Rust item.
    pub const THIRD: f32 = 1.0 / 3.0;
    pub const PRODUCT_F: f32 = 0.1 * 0.3;
    pub const SUM_F: f32 = 0.1 + 0.2;
    pub const CHAIN_F: f32 = THIRD * 3.0 + SUM_F;
    pub const NEG_F: f32 = -(0.1 + 0.2);
    pub const ABSORB: f32 = 16777216.0 + 1.0;
    pub const DIFF_F: f32 = 1.0 - 0.9;
    pub const QUARTER: f32 = 10.0 / 4.0 / 10.0;

    // `bool`: `!`, `&&`, `||` and parentheses.
    pub const YES: bool = true;
    pub const NO: bool = false;
    pub const NOT_NO: bool = !NO;
    pub const AND_C: bool = YES && !NO;
    pub const OR_C: bool = NO || (YES && NO);
    pub const MIXED_B: bool = !(YES && NO) && (NO || YES);

    fn flag(b: bool) -> u32 {
        if b { 1u32 } else { 0u32 }
    }

    #[kernel(workgroup_size(1, 1, 1))]
    pub fn read_all(
        id: SV_DispatchThreadID,
        mut uvals: RWStructuredBuffer<u32>,
        mut ivals: RWStructuredBuffer<i32>,
        mut fvals: RWStructuredBuffer<f32>,
        mut flags: RWStructuredBuffer<u32>,
    ) {
        if id.x == 0u32 {
            uvals[0] = CELLS;
            uvals[1] = STEP3;
            uvals[2] = HIGH;
            uvals[3] = PRODUCT;
            uvals[4] = QUOT;
            uvals[5] = REM;
            uvals[6] = SHL;
            uvals[7] = SHL_TOP;
            uvals[8] = HALF_TOP;
            uvals[9] = SHR;
            uvals[10] = NOT;
            uvals[11] = BITS;
            uvals[12] = MIXED;
            uvals[13] = NOT_HALF;
            uvals[14] = NOT_REM;
            uvals[15] = NOT_SHR;
            ivals[0] = NEG_PROD;
            ivals[1] = I_MIN;
            ivals[2] = TRUNC_DIV;
            ivals[3] = TRUNC_REM;
            ivals[4] = DIV_NEG;
            ivals[5] = REM_NEG;
            ivals[6] = ASH;
            ivals[7] = ISHL;
            ivals[8] = INOT;
            ivals[9] = INOT_NEG;
            ivals[10] = SPAN;
            ivals[11] = SHL_DIV;
            ivals[12] = SHL_SHR;
            fvals[0] = THIRD;
            fvals[1] = PRODUCT_F;
            fvals[2] = SUM_F;
            fvals[3] = CHAIN_F;
            fvals[4] = NEG_F;
            fvals[5] = ABSORB;
            fvals[6] = DIFF_F;
            fvals[7] = QUARTER;
            flags[0] = flag(NOT_NO);
            flags[1] = flag(AND_C);
            flags[2] = flag(OR_C);
            flags[3] = flag(MIXED_B);
        }
    }
}

/// What the host reads from the same items, in the order the kernel writes them (never a copied number).
struct ExprExpected {
    uvals: Vec<u32>,
    ivals: Vec<i32>,
    fvals: Vec<f32>,
    flags: Vec<u32>,
}

fn expr_expected() -> ExprExpected {
    use cexpr::*;
    ExprExpected {
        uvals: vec![
            CELLS, STEP3, HIGH, PRODUCT, QUOT, REM, SHL, SHL_TOP, HALF_TOP, SHR, NOT, BITS, MIXED,
            NOT_HALF, NOT_REM, NOT_SHR,
        ],
        ivals: vec![
            NEG_PROD, I_MIN, TRUNC_DIV, TRUNC_REM, DIV_NEG, REM_NEG, ASH, ISHL, INOT, INOT_NEG,
            SPAN, SHL_DIV, SHL_SHR,
        ],
        fvals: vec![
            THIRD, PRODUCT_F, SUM_F, CHAIN_F, NEG_F, ABSORB, DIFF_F, QUARTER,
        ],
        flags: [NOT_NO, AND_C, OR_C, MIXED_B]
            .into_iter()
            .map(u32::from)
            .collect(),
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn the_expression_constants_hold_the_values_rustc_computes() {
    // Pins the Rust side of the comparison: the expectation reads these items, and these are what rustc computed.
    let e = expr_expected();
    assert_eq!(cexpr::CELLS, 4096);
    assert_eq!(cexpr::STEP3, 16 * 16 - 5);
    assert_eq!(cexpr::HIGH, 4_000_000_000);
    assert!(cexpr::HIGH > i32::MAX as u32);
    assert_eq!(cexpr::HALF_TOP, 1 << 30);
    assert_eq!(cexpr::I_MIN, i32::MIN);
    assert_eq!((cexpr::TRUNC_DIV, cexpr::TRUNC_REM), (-3, -1));
    assert_eq!((cexpr::DIV_NEG, cexpr::REM_NEG), (-3, 1));
    assert_eq!(cexpr::ASH, -4);
    assert_eq!(cexpr::INOT, -6);
    assert_eq!(cexpr::ABSORB, 16777216.0);
    assert_eq!(cexpr::NOT_HALF, 2147483615);
    assert_eq!(cexpr::NOT_SHR, 2147483615);
    assert_eq!(cexpr::NOT_REM, (u32::MAX - 64) % 7);
    assert_eq!((cexpr::SHL_DIV, cexpr::SHL_SHR), (-1073741824, -536870912));
    assert_eq!(e.flags, [1, 1, 0, 1]);
}

#[test]
fn expression_constants_read_the_same_values_on_the_gpu_and_the_twin() {
    let want = expr_expected();

    let device = common::device();
    let info = device.adapter_info();
    eprintln!("adapter: {} backend: {:?}", info.name, info.backend);

    let uvals = device.create_typed_buffer(
        "cexpr uvals",
        &vec![0u32; want.uvals.len()],
        GpuBufferAccess::ReadWrite,
    );
    let ivals = device.create_typed_buffer(
        "cexpr ivals",
        &vec![0i32; want.ivals.len()],
        GpuBufferAccess::ReadWrite,
    );
    let fvals = device.create_typed_buffer(
        "cexpr fvals",
        &vec![0f32; want.fvals.len()],
        GpuBufferAccess::ReadWrite,
    );
    let flags = device.create_typed_buffer(
        "cexpr flags",
        &vec![0u32; want.flags.len()],
        GpuBufferAccess::ReadWrite,
    );
    device
        .dispatch_buffers(
            &cexpr::read_all::DESCRIPTOR,
            1,
            &[
                BufferBinding::read_write(&uvals).independent_length(),
                BufferBinding::read_write(&ivals).independent_length(),
                BufferBinding::read_write(&fvals).independent_length(),
                BufferBinding::read_write(&flags).independent_length(),
            ],
        )
        .unwrap();
    let gpu_uvals = device.read_typed_buffer(&uvals).unwrap();
    let gpu_ivals = device.read_typed_buffer(&ivals).unwrap();
    let gpu_fvals = device.read_typed_buffer(&fvals).unwrap();
    let gpu_flags = device.read_typed_buffer(&flags).unwrap();
    assert_eq!(gpu_uvals, want.uvals, "gpu: u32 expression constants");
    assert_eq!(gpu_ivals, want.ivals, "gpu: i32 expression constants");
    assert_eq!(
        bits(&gpu_fvals),
        bits(&want.fvals),
        "gpu: f32 expression constants, bit for bit"
    );
    assert_eq!(gpu_flags, want.flags, "gpu: bool expression constants");
    eprintln!(
        "cexpr: u32={} i32={} f32_bit_equal={} bool={}",
        want.uvals.len(),
        want.ivals.len(),
        want.fvals.len(),
        want.flags.len()
    );

    // The twin: a second assertion, never GPU evidence.
    let cpu = CpuDevice::new();
    let t_uvals = cpu.buffer(&vec![0u32; want.uvals.len()]);
    let t_ivals = cpu.buffer(&vec![0i32; want.ivals.len()]);
    let t_fvals = cpu.buffer(&vec![0f32; want.fvals.len()]);
    let t_flags = cpu.buffer(&vec![0u32; want.flags.len()]);
    cexpr::read_all::dispatch(
        &cpu,
        1,
        Independent(&t_uvals),
        Independent(&t_ivals),
        Independent(&t_fvals),
        Independent(&t_flags),
    )
    .unwrap();
    assert_eq!(t_uvals.read(), want.uvals, "twin: u32 expression constants");
    assert_eq!(t_ivals.read(), want.ivals, "twin: i32 expression constants");
    assert_eq!(
        bits(&t_fvals.read()),
        bits(&want.fvals),
        "twin: f32 expression constants, bit for bit"
    );
    assert_eq!(
        t_flags.read(),
        want.flags,
        "twin: bool expression constants"
    );
}
