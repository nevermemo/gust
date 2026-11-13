//! The collision table behind collision-only identifier renaming (M6 plan 2.5, R1.1).
//!
//! This file is plain data and string templates with no `crate::` paths, because two crates read it: the macro
//! (`slang/mod.rs` declares it as a module) and the Slangc-tier table test (`crates/gust/tests/diagnostics.rs`
//! includes it with `#[path]`). Do not add a dependency.
//!
//! A user name that is in [`COLLISIONS`] is emitted as `__gust_r_<name>` (the pure function [`emit_name`]); every
//! other name is emitted unchanged. The prefix is inside the space E0901 denies to users, so a renamed name cannot
//! collide with a user name, and removing the prefix where an emitted name starts (a word boundary, in front of a name) from a slangc message gives back the Rust spelling.
//!
//! An entry is in the table for one or both of two reasons, first measured on slangc 2026.13.1 (D46) and checked again
//! on 2026.20 (slice m4.z, 2026-11-02: the table tests of `crates/gust/tests/diagnostics.rs` passed unchanged, 24
//! tests); the Slangc tier, which runs that test file, passed on the pinned 2026.21 (slice m8.p):
//!
//! - **class 1, `fails_in` not empty:** the bare name makes slangc fail in the listed positions (S6.3, S6.3b:
//!   `plans/m6/probe/kw/kw_results.json` and `kw_results3.json`, both targets agreeing). The table test compiles the
//!   bare name at every position and requires exactly the listed set to fail.
//! - **class 2, `emitted`:** GUST's own generated text uses the name (a type, a constructor-style cast, a call, `none`
//!   for `None`), so a user name of that spelling would capture it. Most captures fail loudly; `true` and `false` capture silently (the program compiles and means the other value), which the table test pins by comparing the generated code, and `var` breaks every later local. Measured where the entry says so; the S6.3b
//!   additions (`StructuredBuffer`, `RWStructuredBuffer`, `ConstantBuffer`) are in the struct position.
//!
//! Class 3 (reserved for later measurements) is empty: every Rust keyword that is not in the table, spelled as a raw
//! identifier, compiled in all six positions (probe of 2026-11-01 over 46 keywords, `else`, `loop`, `match`, `fn` and
//! `let` among them). A later slangc version may add an entry only with a measurement attached.

use std::borrow::Cow;

/// The text put in front of a renamed name.
pub const PREFIX: &str = "__gust_r_";

/// Where an identifier can be declared in emitted Slang.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Position {
    /// A `let` binding.
    Local,
    /// A struct field, at the declaration, an access and a struct literal.
    Field,
    /// A helper function, at the declaration and the call.
    Helper,
    /// A struct name.
    Struct,
    /// A kernel parameter, which is emitted as a global resource name.
    Global,
    /// A helper function parameter.
    HelperParameter,
}

#[cfg(test)]
impl Position {
    pub const ALL: [Position; 6] = [
        Position::Local,
        Position::Field,
        Position::Helper,
        Position::Struct,
        Position::Global,
        Position::HelperParameter,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Position::Local => "local",
            Position::Field => "field",
            Position::Helper => "helper",
            Position::Struct => "struct",
            Position::Global => "global",
            Position::HelperParameter => "helper parameter",
        }
    }

    /// The label the keyword probes (S6.3, S6.3b) used for this position.
    pub fn probe_label(self) -> &'static str {
        match self {
            Position::Local => "local",
            Position::Field => "field",
            Position::Helper => "func",
            Position::Struct => "struct",
            Position::Global => "global",
            Position::HelperParameter => "hparam",
        }
    }
}

/// One table entry.
// The fields are read by the table tests, not by the macro.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug)]
pub struct Collision {
    pub name: &'static str,
    /// Class 1: the positions in which the bare name fails slangc. Empty for a class-2-only entry.
    pub fails_in: &'static [Position],
    /// Class 2: GUST's emitted text uses this name.
    pub emitted: bool,
}

use Position::{Field, Global, Helper, HelperParameter, Local, Struct};

const fn hard(name: &'static str, fails_in: &'static [Position]) -> Collision {
    Collision {
        name,
        fails_in,
        emitted: false,
    }
}

const fn emitted(name: &'static str) -> Collision {
    Collision {
        name,
        fails_in: &[],
        emitted: true,
    }
}

const fn both(name: &'static str, fails_in: &'static [Position]) -> Collision {
    Collision {
        name,
        fails_in,
        emitted: true,
    }
}

/// The table, sorted by name (byte order) and without duplicates; a test pins both.
pub const COLLISIONS: &[Collision] = &[
    // Struct position: GUST declares structs before the resource globals it names (`Atomic<T>` is the element
    // type of an atomic buffer). Measured 2026-11-01 (`struct Atomic` then `RWStructuredBuffer<Atomic<uint>>`:
    // E39999 "expected a generic").
    emitted("Atomic"),
    // Struct position: `ConstantBuffer<T>` is emitted for a uniform. Measured 2026-11-01 (the S6.3b follow-up left
    // it untested): a struct of that name breaks a `ConstantBuffer<uint>` global after it.
    emitted("ConstantBuffer"),
    // The generic type behind `Option<T>`.
    emitted("Optional"),
    // Struct position (S6.3b: a struct of that name breaks every `RWStructuredBuffer<T>` global after it; the position
    // templates declare one, so the bare name also fails there, but the record is the capture template's).
    emitted("RWStructuredBuffer"),
    // Struct position (S6.3b).
    emitted("StructuredBuffer"),
    hard("This", &[Field]),
    // The generic type behind a tuple `(A, B)` (slice m9.t).
    emitted("Tuple"),
    // Also denied to users by E0901 (`__Gust`), so it is unreachable and has no capture template; listed because the
    // plan names it.
    emitted("__GustResult"),
    hard("__target_switch", &[Struct, Global]),
    // `bool(x)`, `int(x)`, `uint(x)` and `float(x)` are the emitted casts and literal spellings.
    emitted("bool"),
    hard("break", &[Struct, Global]),
    hard("case", &[Struct, Global]),
    hard("class", &[Struct, Global]),
    hard("continue", &[Struct, Global]),
    hard("default", &[Struct, Global]),
    hard("defer", &[Struct, Global]),
    hard("discard", &[Struct, Global]),
    hard("do", &[Struct, Global]),
    // M6 (`Vec4::dot`).
    emitted("dot"),
    hard("enum", &[Struct, Global]),
    // The boolean literals: a local `r#false` or `r#true` is a variable, not a literal, to every `true` and `false`
    // GUST writes after it, and slangc takes the program with the other value (measured 2026-11-01: `var true = false;
    // var x = true;` stores 9 where Rust gives 7). A silent change, not a failure, so the capture test compares code.
    emitted("false"),
    both("float", &[Struct]),
    both("float2", &[Helper]),
    both("float2x2", &[Helper]),
    both("float3", &[Helper]),
    both("float3x3", &[Helper]),
    both("float4", &[Helper]),
    both("float4x4", &[Helper]),
    hard("for", &[Struct, Global]),
    hard("if", &[Struct, Global]),
    emitted("int"),
    // M6 emits `int2` and `int4` for `IVec2` and `IVec4` (R1-A.4: `ok` in all four S6.3 positions today).
    emitted("int2"),
    emitted("int4"),
    // The constructor of a tuple value `(a, b)` (slice m9.t).
    emitted("makeTuple"),
    // Slang's generic `matrix<T, R, C>`, in case the M6 lowering needs it.
    emitted("matrix"),
    // M6 (`Mat4 * Vec4` helpers).
    emitted("mul"),
    hard("new", &[Local, Helper, Global, HelperParameter]),
    hard("no_diff", &[Struct, Global]),
    // `None` is emitted as `none`; a local `none` of another type breaks it (measured 2026-11-01).
    emitted("none"),
    hard("operator", &[Local, Field, Helper, Global, HelperParameter]),
    hard("return", &[Struct, Global]),
    hard("spirv_asm", &[Local, Helper, Global, HelperParameter]),
    hard("struct", &[Struct, Global]),
    hard("switch", &[Struct, Global]),
    // The receiver of a method is written `this`. A parameter or local of that name inside a method would take its
    // place and compile (`this.a - this.a`), so the bare name is a silent capture, not a failure at a declaration
    // (measured 2026-11-09 on slangc 2026.21: every position takes it; the member probe `capture_source` fails).
    emitted("this"),
    // M6 (`Mat4::transpose`).
    emitted("transpose"),
    // See `false`.
    emitted("true"),
    hard("try", &[Struct]),
    hard("typealias", &[Struct, Global]),
    hard("typedef", &[Struct, Global]),
    emitted("uint"),
    // M6 emits `uint2` and `uint4` for `UVec2` and `UVec4`.
    emitted("uint2"),
    // The invocation id is `uint3` today: class 1 in the struct and global positions, and emitted.
    both("uint3", &[Struct, Global]),
    emitted("uint4"),
    // Every local GUST writes is `var name = ..`; a local named `var` breaks all later ones (measured 2026-11-01:
    // E30060 on both targets, and the same for a helper, struct or global of that name).
    emitted("var"),
    // Slang's generic `vector<T, N>`, in case the M6 lowering needs it.
    emitted("vector"),
    hard("void", &[Struct, Global]),
    hard("while", &[Struct, Global]),
];

/// The interfaces the standard module of the pinned slangc declares (slice m9.g1), read from the interface declarations
/// embedded in `bin/slang-compiler.dll` of slangc 2026.21 (the same 46 names in 2026.22, m9.p) (`grep -a -o -E "interface +(I[A-Z][A-Za-z0-9]+|__[A-Za-z0-9_]+)"`;
/// `IFoo` is a doc comment's example and is left out). A user `trait IFloat` becomes `interface IFloat` next to the
/// standard module's own. Measured on 2026.21 (WGSL, slice m9.g1): a user interface of each of nine of these names
/// (`IFloat`, `IInteger`, `IComparable`, `IArithmetic`, `ILogical`, `IDefaultInitializable`, `IFunc`, `IArray`,
/// `IRWArray`) with a struct and an enum conforming still compiles, the user's declaration shadowing the standard one,
/// so none is a class-1 failure; they are renamed like the table's names so that no later use of a bound (`T : IFloat`
/// of a generic helper) can reach the standard interface, and a slangc message names the Rust spelling again after
/// `strip_rename_prefix`. Sorted by byte and without duplicates, as the table is (a test pins both). Not class 1 or 2:
/// the Slangc-tier table tests do not read this list.
pub const STDLIB_INTERFACES: &[&str] = &[
    "IArithmetic",
    "IArithmeticAtomicable",
    "IArray",
    "IAtomicable",
    "IBackwardDifferentiable",
    "IBitAtomicable",
    "IBufferDataLayout",
    "IBwdCallable",
    "IComparable",
    "ICoopElement",
    "IDefaultInitializable",
    "IDiffTensorWrapper",
    "IDifferentiable",
    "IDifferentiableArithmetic",
    "IDifferentiableFunc",
    "IDifferentiableMutatingFunc",
    "IDifferentiablePtrType",
    "IFloat",
    "IFloatingPointCoopElement",
    "IForwardDifferentiable",
    "IFunc",
    "IInteger",
    "ILogical",
    "IMutatingFunc",
    "IOpaqueDescriptor",
    "IPhysicalBuffer",
    "IRWArray",
    "IRWPhysicalBuffer",
    "IRangedValue",
    "ITexelElement",
    "__BuiltinArithmeticType",
    "__BuiltinFloatingPointType",
    "__BuiltinInt32Type",
    "__BuiltinInt64Type",
    "__BuiltinIntegerType",
    "__BuiltinLogicalType",
    "__BuiltinRealType",
    "__BuiltinSamplerFeedbackType",
    "__BuiltinScalarType",
    "__BuiltinSignedArithmeticType",
    "__BuiltinType",
    "__EnumType",
    "__IDynamicResourceCastable",
    "__IMinMax",
    "__ITextureShape",
    "__ITextureShape1D2D3D",
];

/// Whether `text` (already without a raw prefix) is in the table or names an interface of Slang's standard module.
pub fn collides(text: &str) -> bool {
    COLLISIONS
        .binary_search_by(|entry| entry.name.cmp(text))
        .is_ok()
        || STDLIB_INTERFACES.binary_search(&text).is_ok()
}

/// The Slang spelling of a user name: the name itself, or `__gust_r_<name>` when it is in the table or names an interface
/// of the standard module.
///
/// A pure function of the text, so every site agrees without scope analysis. It is injective because user names that
/// start with `__gust_` are rejected (E0901), and the inverse is "delete the prefix where an emitted name starts" (`strip_rename_prefix` in `crates/gust/src/slang/mod.rs`).
pub fn emit_name(text: &str) -> Cow<'_, str> {
    if collides(text) {
        Cow::Owned(format!("{PREFIX}{text}"))
    } else {
        Cow::Borrowed(text)
    }
}

/// The text of a small kernel that declares `name` in `position`, in the shapes GUST emits (the probe's shapes of
/// S6.3 and S6.3b, so a recorded failure is the failure of exactly this text). `name` is written verbatim: pass the
/// bare name to measure a collision, the renamed one to check that it compiles. The entry point is `gpu_table_run`.
#[cfg(test)]
#[allow(dead_code)] // the Slangc-tier tests use it, the macro's Unit tests do not
pub fn position_source(position: Position, name: &str) -> String {
    const KERNEL: &str = "[shader(\"compute\")]\n[numthreads(64, 1, 1)]\nvoid gpu_table_run(uint3 gid : SV_DispatchThreadID)\n";
    match position {
        Position::Local => {
            format!(
                "RWStructuredBuffer<float> ob;\n{KERNEL}{{ float {name} = 1.0; ob[gid.x] = {name}; }}\n"
            )
        }
        Position::Field => format!(
            "struct S {{ float {name}; }};\nRWStructuredBuffer<S> ob;\n{KERNEL}{{ S s; s.{name} = 1.0; ob[gid.x] = s; }}\n"
        ),
        Position::Helper => format!(
            "float {name}(float v) {{ return v + 1.0; }}\nRWStructuredBuffer<float> ob;\n{KERNEL}{{ ob[gid.x] = {name}(2.0); }}\n"
        ),
        Position::Struct => format!(
            "struct {name} {{ float v; }};\nRWStructuredBuffer<float> ob;\n{KERNEL}{{ {name} s; s.v = 1.0; ob[gid.x] = s.v; }}\n"
        ),
        Position::Global => {
            format!("RWStructuredBuffer<float> {name};\n{KERNEL}{{ {name}[gid.x] = 1.0; }}\n")
        }
        Position::HelperParameter => format!(
            "float hf(float {name}) {{ return {name} + 1.0; }}\nRWStructuredBuffer<float> ob;\n{KERNEL}{{ ob[gid.x] = hf(2.0); }}\n"
        ),
    }
}

/// The Rust names that `emit_type` maps to Slang types (the scalars, the vectors and the matrices) before it falls back
/// to `emit_name`. E0203 rejects a user struct of one of these names (before m6.m1 such a struct was declared under the
/// renamed spelling but every use of it emitted the builtin, which slangc refused), so the struct position of these
/// names is not reachable and not exercised end to end.
#[cfg(test)]
#[allow(dead_code)] // the macro's Unit tests use it, the Slangc-tier tests do not
pub const MAPPED_TYPE_NAMES: &[&str] = &[
    "bool", "float", "int", "uint", "Vec2", "Vec3", "Vec4", "IVec2", "IVec4", "UVec2", "UVec3",
    "UVec4", "Mat2", "Mat3", "Mat4",
];

/// For a class-2 entry, the text of a kernel that declares `spelled` (the bare name to measure the capture, the
/// renamed one to check that the rename removes it) next to the construct GUST emits with that name. `None` for an
/// entry that cannot be exercised: `__GustResult`, which E0901 denies to users. Measured 2026-11-01 on slangc
/// 2026.13.1 and again on the pinned 2026.20 on 2026-11-02 (the Slangc-tier capture tests): with the bare name every
/// one of these fails on both targets, except `true` and `false`, which compile and mean the other value (see
/// [`SILENT_CAPTURES`]); with the renamed name every one compiles.
#[cfg(test)]
#[allow(dead_code)] // the Slangc-tier tests use it, the macro's Unit tests do not
pub fn capture_source(entry: &str, spelled: &str) -> Option<String> {
    const KERNEL: &str = "[shader(\"compute\")]\n[numthreads(64, 1, 1)]\nvoid gpu_table_run(uint3 gid : SV_DispatchThreadID)\n";
    let n = spelled;
    // A user local, then the construct.
    let local = |use_it: &str| {
        format!("RWStructuredBuffer<uint> ob;\n{KERNEL}{{ float {n} = 1.0; {use_it} }}\n")
    };
    // A user `bool` local of the given value, then the construct.
    let bool_local = |value: &str, use_it: &str| {
        format!(
            "RWStructuredBuffer<uint> ob;
{KERNEL}{{ bool {n} = {value}; ob[gid.x] = {use_it}; }}
"
        )
    };
    // A user struct declared before the globals that name the emitted type, as GUST orders its output.
    let in_struct = |setup: &str, use_it: &str| {
        format!(
            "struct {n} {{ float v; }};\n{setup}RWStructuredBuffer<uint> ob;\n{KERNEL}{{ {use_it} }}\n"
        )
    };
    Some(match entry {
        "Atomic" => in_struct(
            "RWStructuredBuffer<Atomic<uint>> ab;\n",
            "ab[gid.x].add(1u);",
        ),
        "ConstantBuffer" => in_struct(
            "[[vk::binding(0, 0)]] ConstantBuffer<uint> cb;\n",
            "ob[gid.x] = cb;",
        ),
        "Optional" => in_struct(
            "",
            "Optional<uint> o = none; ob[gid.x] = o.hasValue ? 1u : 0u;",
        ),
        "RWStructuredBuffer" => in_struct(
            "RWStructuredBuffer<float> rb;\n",
            "rb[gid.x] = 1.0; ob[gid.x] = 1u;",
        ),
        "StructuredBuffer" => in_struct(
            "StructuredBuffer<float> rb;\n",
            "ob[gid.x] = uint(rb[gid.x]);",
        ),
        // A user struct named `Tuple` hides Slang's generic one; a user local named `makeTuple` hides its constructor.
        "Tuple" => in_struct(
            "",
            "Tuple<uint, float> t = makeTuple(1u, 2.0); ob[gid.x] = t._0;",
        ),
        "makeTuple" => local("Tuple<uint, float> t = makeTuple(1u, 2.0); ob[gid.x] = t._0;"),
        "bool" => local("bool b = bool(ob[gid.x]); ob[gid.x] = b ? 1u : 0u;"),
        "int" => local("int i = int(ob[gid.x]); ob[gid.x] = uint(i);"),
        "uint" => local("ob[gid.x] = uint(2.0);"),
        "float" => local("ob[gid.x] = uint(float(3));"),
        "float2" => local("float2 a = float2(1.0, 2.0); ob[gid.x] = uint(a.x);"),
        "float3" => local("float3 a = float3(1.0, 2.0, 3.0); ob[gid.x] = uint(a.x);"),
        "float4" => local("float4 a = float4(1.0, 2.0, 3.0, 4.0); ob[gid.x] = uint(a.x);"),
        "float2x2" => local("float2x2 a = float2x2(1.0); ob[gid.x] = uint(a[0][0]);"),
        "float3x3" => local("float3x3 a = float3x3(1.0); ob[gid.x] = uint(a[0][0]);"),
        "float4x4" => local("float4x4 a = float4x4(1.0); ob[gid.x] = uint(a[0][0]);"),
        "dot" => local("ob[gid.x] = uint(dot(float3(1.0), float3(2.0)));"),
        "mul" => local("ob[gid.x] = uint(mul(float2x2(1.0), float2(2.0)).x);"),
        "transpose" => local("float2x2 t = transpose(float2x2(1.0)); ob[gid.x] = uint(t[0][0]);"),
        "int2" => local("int2 a = int2(1, 2); ob[gid.x] = uint(a.x);"),
        "int4" => local("int4 a = int4(1, 2, 3, 4); ob[gid.x] = uint(a.x);"),
        "uint2" => local("uint2 a = uint2(1u, 2u); ob[gid.x] = a.x;"),
        "uint3" => local("uint3 a = uint3(1u, 2u, 3u); ob[gid.x] = a.x;"),
        "uint4" => local("uint4 a = uint4(1u, 2u, 3u, 4u); ob[gid.x] = a.x;"),
        "matrix" => {
            local("matrix<float, 2, 2> m = matrix<float, 2, 2>(1.0); ob[gid.x] = uint(m[0][0]);")
        }
        "vector" => local("vector<float, 3> v = vector<float, 3>(1.0); ob[gid.x] = uint(v.x);"),
        "none" => local("Optional<uint> o = none; ob[gid.x] = o.hasValue ? 1u : 0u;"),
        // `true` and `false` captured by a local of the other value: see [`SILENT_CAPTURES`].
        "true" => bool_local("false", "true ? 7u : 9u"),
        "false" => bool_local("true", "false ? 7u : 9u"),
        "var" => local("var y = 2.0; ob[gid.x] = uint(y);"),
        // The receiver of a member function, captured by a parameter of that name: `this.a` is then a member of a float.
        "this" => format!(
            "struct S {{ float a; float f(float {n}) {{ return this.a - {n}; }} }};
RWStructuredBuffer<S> sb;
RWStructuredBuffer<float> ob;
{KERNEL}{{ ob[gid.x] = sb[gid.x].f(1.0); }}
"
        ),
        _ => return None,
    })
}

/// The class-2 entries whose capture does not fail: the program compiles and means something else. For each, the
/// bare compile must hold the first text and not the second, and the renamed compile the second and not the first
/// (the WGSL constants of [`capture_source`], measured 2026-11-01: slangc folds `true ? 7u : 9u`).
#[cfg(test)]
#[allow(dead_code)] // the Slangc-tier tests use it, the macro's Unit tests do not
pub const SILENT_CAPTURES: &[(&str, &str, &str)] =
    &[("false", "u32(7)", "u32(9)"), ("true", "u32(9)", "u32(7)")];

/// The class-1 record of the keyword probes, copied from `plans/m6/probe/kw/kw_results.json` (S6.3: local, field,
/// func, struct) and `kw_results3.json` (S6.3b: global, hparam), with the probe's own position labels. It is a
/// snapshot of a measurement and is not edited together with [`COLLISIONS`]: a test requires every name here to be in
/// the table with exactly these positions, so dropping or mis-recording an entry fails. `id` (a warning that came
/// from the probe's own kernel parameter) and the garbage keys of the unsplit string in `kw_results2.json` are not
/// names and are left out.
#[cfg(test)]
pub const MEASURED_CLASS1: &[(&str, &[&str])] = &[
    ("This", &["field"]),
    ("__target_switch", &["struct", "global"]),
    ("break", &["struct", "global"]),
    ("case", &["struct", "global"]),
    ("class", &["struct", "global"]),
    ("continue", &["struct", "global"]),
    ("default", &["struct", "global"]),
    ("defer", &["struct", "global"]),
    ("discard", &["struct", "global"]),
    ("do", &["struct", "global"]),
    ("enum", &["struct", "global"]),
    ("float", &["struct"]),
    ("float2", &["func"]),
    ("float2x2", &["func"]),
    ("float3", &["func"]),
    ("float3x3", &["func"]),
    ("float4", &["func"]),
    ("float4x4", &["func"]),
    ("for", &["struct", "global"]),
    ("if", &["struct", "global"]),
    ("new", &["local", "func", "global", "hparam"]),
    ("no_diff", &["struct", "global"]),
    ("operator", &["local", "field", "func", "global", "hparam"]),
    ("return", &["struct", "global"]),
    ("spirv_asm", &["local", "func", "global", "hparam"]),
    ("struct", &["struct", "global"]),
    ("switch", &["struct", "global"]),
    ("try", &["struct"]),
    ("typealias", &["struct", "global"]),
    ("typedef", &["struct", "global"]),
    ("uint3", &["struct", "global"]),
    ("void", &["struct", "global"]),
    ("while", &["struct", "global"]),
];
