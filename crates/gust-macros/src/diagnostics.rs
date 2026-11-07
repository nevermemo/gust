//! Stable diagnostic codes for every rejection `#[gpu]` reports.
//!
//! `syn::Error` carries only a message, so the code is part of the message text:
//! `[GUST-E0401] <message>`. rustc shows it as `error: [GUST-E0401] <message>`, which
//! makes the code searchable in `docs/DIAGNOSTICS.md`. Codes are a compatibility
//! surface: they never change meaning and are never reused, while the message text may be
//! reworded freely. Renumbering requires a row in `docs/DECISIONS.md`.
//!
//! Families (the hundreds digit):
//!
//! | Family | Area |
//! | --- | --- |
//! | E01xx | module and item shape: inline module, allowed items, structs, signatures, attributes |
//! | E02xx | types: unsupported primitives, casts |
//! | E03xx | expressions and calls: names, built-ins, literals, paths, bindings |
//! | E04xx | control flow and loops |
//! | E05xx | `Option` and `Result` |
//! | E06xx | atomics |
//! | E07xx | macro entry and `#[kernel]` options |
//! | E08xx | emitter-only rejections |
//! | E09xx | reserved names and shadowing |
//!
//! Every error the crate creates goes through [`error`] (or [`wrap`] for a `syn` parse
//! error), and every `syn` parse goes through [`parse2`], [`parse_args_with`] or
//! [`parse_nested_meta`], which wrap the failure. The workspace `clippy.toml` bars
//! `syn::Error::new`, `new_spanned`, the `syn` parse functions and `parse_macro_input!`
//! (`disallowed-methods`, `disallowed-macros`); this file is the only one that allows them,
//! and CI runs clippy with `-D warnings`. A test keeps the allowance out of other files.

// This module is where `syn` errors are made and `syn` input is parsed, so the items that
// `clippy.toml` bars everywhere else are allowed here. Do not add this allowance elsewhere:
// `only_this_file_allows_the_barred_syn_items` fails if another source file names it.
#![allow(clippy::disallowed_methods)]

use std::fmt::Display;

use quote::ToTokens;

/// Declares [`Code`], its stable text and `ALL` (a list of every code, compiled for tests only) from one list, so a
/// code cannot be added to the enum without being listed.
macro_rules! codes {
    ($($(#[$doc:meta])* $name:ident = $number:literal => $text:literal;)*) => {
        /// A stable diagnostic code. The discriminant is the number in the text.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[repr(u16)]
        pub(crate) enum Code {
            $($(#[$doc])* $name = $number,)*
        }

        impl Code {
            /// The code as users see it, for example `"GUST-E0401"`.
            pub(crate) fn text(self) -> &'static str {
                match self {
                    $(Self::$name => $text,)*
                }
            }
        }

        /// Every code, in declaration order (the tests walk it; nothing else reads it).
        #[cfg(test)]
        pub(crate) const ALL: &[Code] = &[$(Code::$name,)*];
    };
}

codes! {
    // E01xx: module and item shape.
    /// `#[gpu]` on a module without inline content.
    InlineModuleRequired = 101 => "GUST-E0101";
    /// An item other than a function or a plain struct inside the module.
    UnsupportedModuleItem = 102 => "GUST-E0102";
    /// A struct with generic parameters or a `where` clause.
    GenericStruct = 103 => "GUST-E0103";
    /// A tuple or unit struct.
    StructNeedsNamedFields = 104 => "GUST-E0104";
    /// A struct without fields.
    EmptyStruct = 105 => "GUST-E0105";
    /// A hand-written `#[repr(..)]` on a GPU struct.
    ManualRepr = 106 => "GUST-E0106";
    /// A hand-written `Clone` or `Copy` derive on a GPU struct.
    ManualCloneCopy = 107 => "GUST-E0107";
    /// A `#[derive(..)]` argument list that does not parse.
    MalformedDerive = 108 => "GUST-E0108";
    /// A kernel without exactly one dispatch-thread parameter.
    KernelDispatchParameter = 109 => "GUST-E0109";
    /// A kernel with a return type other than unit.
    KernelReturnsUnit = 110 => "GUST-E0110";
    /// A helper function taking a buffer or uniform parameter.
    HelperResourceParameter = 111 => "GUST-E0111";
    /// An `async` function.
    AsyncFunction = 112 => "GUST-E0112";
    /// An `unsafe` function.
    UnsafeFunction = 113 => "GUST-E0113";
    /// An `extern` or generic function.
    ExternOrGenericFunction = 114 => "GUST-E0114";
    /// An attribute with no proven Rust-to-Slang meaning.
    UnsupportedAttribute = 115 => "GUST-E0115";
    /// A `self` parameter.
    SelfParameter = 116 => "GUST-E0116";
    /// A parameter whose pattern is not an identifier.
    ParameterNeedsIdentifier = 117 => "GUST-E0117";
    /// A POD struct without named fields, found while generating the host side.
    PodStructNamedFields = 118 => "GUST-E0118";
    /// A `Workgroup<..>` parameter on a function that is not a kernel.
    WorkgroupParameterOnHelper = 119 => "GUST-E0119";

    // E02xx: types.
    /// A primitive type outside the 32-bit scalar subset.
    UnsupportedPrimitive = 201 => "GUST-E0201";
    /// A cast to a type other than `f32`, `i32` or `u32` (or their aliases).
    UnsupportedCastTarget = 202 => "GUST-E0202";
    /// A struct named like a type the dialect maps itself: a scalar, a vector or a matrix.
    ReservedTypeName = 203 => "GUST-E0203";
    /// A 3-wide vector or a `Mat3` as a struct field or a resource element.
    ThreeWideInBuffer = 204 => "GUST-E0204";
    /// A `Mat2` or a `bool` in a struct that a uniform reaches: neither has a constant-buffer layout a Rust type
    /// matches.
    UniformFieldType = 205 => "GUST-E0205";
    /// A `Uniform<T>` or `ConstantBuffer<T>` parameter whose `T` is not a struct of the module.
    UniformElement = 206 => "GUST-E0206";
    /// A struct reachable from both a uniform parameter and a storage element.
    UniformAndStorageStruct = 207 => "GUST-E0207";
    /// A struct whose fields leave implicit padding, so it cannot be a `GpuPod` type.
    ImplicitPadding = 208 => "GUST-E0208";
    /// A struct that a uniform reaches whose constant-buffer layout has a gap or a tail.
    UniformPadding = 209 => "GUST-E0209";
    /// A `Workgroup<..>` that is not `Workgroup<[T; N]>` with a supported `T` and a nonzero literal `N`, or that
    /// is used anywhere but as a kernel parameter type.
    WorkgroupShape = 210 => "GUST-E0210";

    // E03xx: expressions and calls.
    /// A CPU-only library or type name.
    CpuOnlyName = 301 => "GUST-E0301";
    /// A call to anything but a helper declared in the same module.
    UnknownCall = 302 => "GUST-E0302";
    /// A built-in method called with a turbofish.
    BuiltinTypeArguments = 303 => "GUST-E0303";
    /// A built-in method called with arguments.
    BuiltinArguments = 304 => "GUST-E0304";
    /// A method that is not a supported built-in.
    UnsupportedMethod = 305 => "GUST-E0305";
    /// `.len()` on something other than a storage parameter.
    LenReceiver = 306 => "GUST-E0306";
    /// `.global_id()` on something other than the dispatch-thread parameter.
    GlobalIdReceiver = 307 => "GUST-E0307";
    /// An associated constant or other multi-segment path used as a value.
    ForeignPath = 308 => "GUST-E0308";
    /// A `const { .. }` block.
    ConstBlock = 309 => "GUST-E0309";
    /// Struct update syntax (`..base`).
    StructUpdateSyntax = 310 => "GUST-E0310";
    /// The `?` operator.
    TryOperator = 311 => "GUST-E0311";
    /// An `unsafe { .. }` block.
    UnsafeBlock = 312 => "GUST-E0312";
    /// A macro invocation inside the module.
    MacroInModule = 313 => "GUST-E0313";
    /// A closure.
    Closure = 314 => "GUST-E0314";
    /// `.await`.
    Await = 315 => "GUST-E0315";
    /// A reference (`&x`, `&mut x`) outside the atomic call form.
    Reference = 316 => "GUST-E0316";
    /// Integer division or remainder by a literal zero.
    DivisionByLiteralZero = 317 => "GUST-E0317";
    /// A `ref`, `ref mut` or `@` binding.
    BindingMode = 318 => "GUST-E0318";
    /// A `let` pattern that is not a single identifier.
    LetNeedsIdentifier = 319 => "GUST-E0319";
    /// `let .. else`.
    LetElse = 320 => "GUST-E0320";
    /// A uniform parameter used as a whole value: `*params`, `let p = params;`, `params` as an argument.
    UniformWholeValue = 321 => "GUST-E0321";
    /// `*=` on a place that contains a call, in a module that names a matrix (the place would be evaluated twice).
    MulAssignPlace = 323 => "GUST-E0323";
    /// A qualified path to a math type or one of its constructors, such as `super::Mat4` or `gust::math::Vec4::new`.
    QualifiedMathPath = 324 => "GUST-E0324";
    /// A method call with arguments whose name is not in the dialect (the accepted methods are listed in the text).
    MethodNotInDialect = 325 => "GUST-E0325";
    /// An unannotated `let` whose initializer is only unsuffixed integer literals (`let x = 0;`, `let y = -(1 + 2);`).
    UnsuffixedLiteralLet = 326 => "GUST-E0326";
    /// `workgroup_barrier(..)` with arguments, or anywhere but a statement of its own.
    BarrierForm = 327 => "GUST-E0327";

    // E04xx: control flow and loops.
    /// A `loop` expression.
    LoopNotSupported = 401 => "GUST-E0401";
    /// A `while` loop.
    WhileNotSupported = 402 => "GUST-E0402";
    /// A loop label or a labelled `break` or `continue`.
    LoopLabel = 403 => "GUST-E0403";
    /// `break` or `continue` outside a `for` loop.
    JumpOutsideLoop = 404 => "GUST-E0404";
    /// `break` or `continue` used as a value.
    JumpAsValue = 405 => "GUST-E0405";
    /// `break` carrying a value.
    BreakWithValue = 406 => "GUST-E0406";
    /// A `mut` loop variable.
    MutableLoopVariable = 407 => "GUST-E0407";
    /// A `for` pattern that is not one identifier or `_`.
    LoopVariablePattern = 408 => "GUST-E0408";
    /// A `for` over anything but an exclusive range.
    ForIterable = 409 => "GUST-E0409";
    /// An inclusive range (`..=`).
    InclusiveRange = 410 => "GUST-E0410";
    /// A range with a missing bound.
    OpenRange = 411 => "GUST-E0411";
    /// An integer-literal loop bound without a type suffix.
    UnsuffixedLoopBound = 412 => "GUST-E0412";
    /// A range used outside the iterable of a `for` loop.
    RangeAsValue = 413 => "GUST-E0413";
    /// `workgroup_barrier()` at a position that not every invocation of the workgroup reaches the same number of
    /// times.
    BarrierPosition = 414 => "GUST-E0414";
    /// `return` in a kernel that uses workgroup memory or a barrier.
    ReturnInBarrierKernel = 415 => "GUST-E0415";
    /// `break` or `continue` in a loop that contains a barrier.
    JumpInBarrierLoop = 416 => "GUST-E0416";

    // E05xx: Option and Result.
    /// `Option` in a struct field or resource element type.
    OptionLayout = 501 => "GUST-E0501";
    /// An `Option` payload that is not a supported scalar or module struct.
    OptionPayload = 502 => "GUST-E0502";
    /// `Result` in a struct field or resource element type.
    ResultLayout = 503 => "GUST-E0503";
    /// A `Result` that is not `Result<T, T>` with a supported payload.
    ResultShape = 504 => "GUST-E0504";
    /// `Some(..)` with other than one argument.
    SomeArguments = 505 => "GUST-E0505";
    /// `Ok(..)` or `Err(..)` with other than one argument.
    OkErrArguments = 506 => "GUST-E0506";
    /// `Some`, `Ok` or `Err` used as a bare value.
    VariantAsValue = 507 => "GUST-E0507";
    /// `is_some`, `is_none`, `is_ok` or `is_err` with arguments.
    PredicateArguments = 508 => "GUST-E0508";
    /// `unwrap_or` with other than one argument.
    UnwrapOrArguments = 509 => "GUST-E0509";
    /// `unwrap` or `expect`.
    UnwrapPanics = 510 => "GUST-E0510";
    /// An `if let` pattern other than `Some(identifier)` or `Ok(identifier)`.
    IfLetPattern = 511 => "GUST-E0511";
    /// An `if let` in a position other than the whole condition.
    IfLetPosition = 512 => "GUST-E0512";
    /// A `Result` match used as a value.
    MatchAsValue = 513 => "GUST-E0513";
    /// A `Result` match statement without its terminating `;`.
    MatchNeedsSemicolon = 514 => "GUST-E0514";
    /// A match on something that is not a known `Result<T, T>`.
    MatchScrutinee = 515 => "GUST-E0515";
    /// An attribute on a match arm.
    MatchArmAttribute = 516 => "GUST-E0516";
    /// A match guard.
    MatchGuard = 517 => "GUST-E0517";
    /// A match arm whose body is not a block.
    MatchArmBlock = 518 => "GUST-E0518";
    /// A match binding that is not a plain identifier.
    MatchBinding = 519 => "GUST-E0519";
    /// A match without exactly one `Ok` arm and one `Err` arm.
    MatchArmCount = 520 => "GUST-E0520";
    /// A match pattern other than `Ok(identifier)` or `Err(identifier)`.
    MatchPattern = 521 => "GUST-E0521";

    // E06xx: atomics.
    /// An atomic operation with the wrong number of arguments.
    AtomicArgumentCount = 601 => "GUST-E0601";
    /// An atomic operation whose first argument is not a reference.
    AtomicNotReference = 602 => "GUST-E0602";
    /// An atomic operation on something other than a read-write buffer parameter.
    AtomicBufferParameter = 603 => "GUST-E0603";
    /// An atomic operation on an element type other than `u32` or `i32`.
    AtomicElementType = 604 => "GUST-E0604";
    /// `atomic_min` or `atomic_max` on a signed element.
    AtomicSignedMinMax = 605 => "GUST-E0605";
    /// An atomic operation used as a value.
    AtomicAsValue = 606 => "GUST-E0606";
    /// Ordinary indexing of a buffer that is also used atomically.
    AtomicDirectAccess = 607 => "GUST-E0607";
    /// An atomic operation whose first argument is `&buffer[index]`, not `&mut`.
    AtomicReferenceNotMutable = 608 => "GUST-E0608";
    /// An atomic operation whose reference is not to an indexed element.
    AtomicOperandNotElement = 609 => "GUST-E0609";
    /// An atomic operation on an element of something other than a named buffer.
    AtomicReceiverNotBuffer = 610 => "GUST-E0610";

    // E07xx: macro entry and `#[kernel]` options.
    /// Arguments on `#[gpu]`.
    GpuArguments = 701 => "GUST-E0701";
    /// `#[gpu]` on something that does not parse as a module.
    ItemNotModule = 702 => "GUST-E0702";
    /// More than one `#[kernel]` on a function.
    DuplicateKernelAttribute = 703 => "GUST-E0703";
    /// `workgroup_size` given twice.
    DuplicateWorkgroupSize = 704 => "GUST-E0704";
    /// `workgroup_size` without exactly three dimensions.
    WorkgroupDimensionCount = 705 => "GUST-E0705";
    /// A workgroup dimension of zero.
    WorkgroupDimensionZero = 706 => "GUST-E0706";
    /// A `#[kernel(..)]` option other than `workgroup_size` and `cpu`.
    UnknownKernelOption = 707 => "GUST-E0707";
    /// A `#[kernel(..)]` argument list that does not parse.
    MalformedKernelAttribute = 708 => "GUST-E0708";
    /// A `cpu` option whose value is not `"unsupported"`.
    UnknownCpuValue = 709 => "GUST-E0709";
    /// `cpu` given twice.
    DuplicateCpuOption = 710 => "GUST-E0710";
    /// A kernel with one file of its precompiled artifact pair on disk, outside export mode.
    HalfArtifactPair = 711 => "GUST-E0711";
    /// Export mode could not write a kernel's records, or refused a record that held CPU twin text.
    ExportWriteFailed = 712 => "GUST-E0712";
    /// Export mode with a `CARGO_PKG_NAME` that cannot name the record directory.
    ExportPackageName = 713 => "GUST-E0713";
    /// `#[gpu]` on a function that is not a map function: no parameter, a return type or a parameter type that is not
    /// a buffer element, or a `#[kernel]` on it.
    MapSignature = 714 => "GUST-E0714";
    /// A map function whose name, or one of whose parameters' names, collides with a name the generated module
    /// defines.
    MapNameCollision = 715 => "GUST-E0715";
    /// `#[gpu]` on a function inside a `#[gpu]` module.
    NestedGpuFunction = 716 => "GUST-E0716";
    /// A kernel with workgroup memory or a barrier that does not say `cpu = "unsupported"`.
    BarrierKernelCpu = 717 => "GUST-E0717";
    /// Workgroup memory of a kernel that adds up to more than 16 KiB.
    WorkgroupMemoryLimit = 718 => "GUST-E0718";
    /// A `par_for!` whose shape is not one of the forms: the loop pattern, the ranges, the binding list, the body, or
    /// an option in front of the loop.
    ParForForm = 719 => "GUST-E0719";
    /// A `par_for!` binding that is named twice, has a type a binding cannot have, or is a `GpuVec` under
    /// `GUST_PROFILE=cpu`.
    ParForBinding = 720 => "GUST-E0720";
    /// A `#[kernel(shape = ..)]` whose value is a name other than `grid` or `pixel`.
    ShapeUnknown = 721 => "GUST-E0721";
    /// A `shape` option given twice, without a value, with a value that is not a bare name, or as `cpu` (a target, not a
    /// shape).
    ShapeOption = 722 => "GUST-E0722";
    /// A pinned shape the kernel's workgroup memory, barrier, invocation built-ins, atomics, storage textures,
    /// outputs or workgroup size rule out.
    ShapeExcluded = 723 => "GUST-E0723";
    /// A pinned shape ruled out by a `mut` invocation parameter or an assignment to it.
    ShapeMutableInvocation = 724 => "GUST-E0724";
    /// A pinned shape ruled out by a write at an index other than the element's own, or by a second or rebound alias
    /// of the own index.
    ShapeOwnIndex = 725 => "GUST-E0725";
    /// A pinned shape ruled out by an output that is not written on every path.
    ShapeConditionalWrite = 726 => "GUST-E0726";
    /// A pinned shape ruled out by a `return` in the kernel body.
    ShapeReturn = 727 => "GUST-E0727";
    /// A pinned shape ruled out by a use of an output other than `out[own] = v` and `out.len()`.
    ShapeOutputUse = 728 => "GUST-E0728";
    /// A construct the kernel's execution shape (its stage profile) does not take.
    ProfileRefusal = 729 => "GUST-E0729";
    // E0730 (a `pixel` pin on a compatible kernel while no `pixel` profile existed) retired with the `pixel` profile
    // (mx.p1a): its number is never reused.

    // E0731-E0736: texture and sampler parameters (mx.t3, D62, D64).
    /// A texture parameter type with a texel or format argument the dialect does not have: a `Texture2D` texel type
    /// outside the nine, a `StorageTexture2D` format that is not read-write, an unknown format, a wrong number of
    /// arguments, or a `Sampler` with arguments.
    TextureType = 731 => "GUST-E0731";
    /// A texture method call with the wrong number of coordinates, or with a coordinate that is visibly not the
    /// right type (a float or negative texel address, an integer `u` or `v`).
    TextureCoordinate = 732 => "GUST-E0732";
    /// A `sample` whose first argument is not a `Sampler` parameter of the kernel.
    SamplerArgument = 733 => "GUST-E0733";
    /// A `sample` on a texture of integer texels, which are never filtered.
    SampleInteger = 734 => "GUST-E0734";
    /// A method a texture parameter does not have: `store` on a sampled texture, `load` on a write-only one,
    /// `sample` on a storage texture, or any other name.
    TextureMethod = 735 => "GUST-E0735";
    /// A texture or sampler parameter used other than as the receiver of a texture method (or the sampler argument of
    /// `sample`), or a texture or sampler type anywhere but a kernel parameter.
    TextureUse = 736 => "GUST-E0736";

    // E0740 to E0750: variants (M8, D139).
    /// A malformed or misplaced `#[variant]`: not `of = <kernel>` with an optional `requires(..)`, on something that is
    /// not a function of a `#[gpu]` module, on a `#[kernel]`, setting an option of the kernel, or with a signature
    /// that is not its kernel's.
    VariantAttribute = 740 => "GUST-E0740";
    /// An unknown capability or alias in a `requires(..)`.
    UnknownCapability = 741 => "GUST-E0741";
    /// A `#[variant(of = name)]` where `name` is no `#[kernel]` function of the module.
    VariantTarget = 742 => "GUST-E0742";
    /// Two bodies of one kernel that require the same set of capabilities, or a set of capabilities that selects
    /// two bodies.
    VariantAmbiguous = 743 => "GUST-E0743";
    /// A profile the crate declares for which a kernel has no body, or the CPU target declared for a kernel with no CPU
    /// form.
    ProfileNotCovered = 744 => "GUST-E0744";
    /// A subgroup operation outside the body of a kernel with `requires(subgroups)`: in a portable body, in a helper,
    /// or in a `par_for!` or pixel form (m8.s, D140).
    SubgroupNotAllowed = 745 => "GUST-E0745";
    /// A kernel over one of the limits of the portable floor, in a crate that sets `limits = "portable"` (m8.f, D143).
    /// For such a crate this code also stands where E0718 would.
    PortableFloor = 746 => "GUST-E0746";
    /// A subgroup operation at a position that is not the whole right-hand side of a `let` or an assignment (or an
    /// operand of pure arithmetic or a comparison there) at the top level of the kernel body or of a literal-range
    /// `for` (m8.s, D140).
    SubgroupPosition = 747 => "GUST-E0747";
    /// No body of a kernel can be used by the profile being built.
    NoBodySelectable = 750 => "GUST-E0750";
    /// A body that uses subgroup operations whose invocation count is not a multiple of 64 (m8.s, D140).
    SubgroupInvocations = 751 => "GUST-E0751";

    // E0753: the crate's declarations in its manifest (M8, D138). E0737 to E0752 belong to the other M8 slices.
    /// A malformed `[package.metadata.gust]` table in the `Cargo.toml` of the crate that holds the `#[gpu]` module, the `#[gpu]` function or the `par_for!` call.
    GustMetadata = 753 => "GUST-E0753";

    // E08xx: rejections only the emitter makes, after validation passed.
    /// A kernel parameter that is neither a resource nor the dispatch thread.
    EntryPointParameter = 801 => "GUST-E0801";
    /// An emitted kernel without a dispatch-thread parameter.
    EntryPointDispatchParameter = 802 => "GUST-E0802";
    /// A `let` without an initializer.
    LetNeedsInitializer = 803 => "GUST-E0803";
    /// An item declared inside a function body.
    NestedItem = 804 => "GUST-E0804";
    /// A macro invocation in statement position.
    MacroStatement = 805 => "GUST-E0805";
    /// A struct literal field without a name.
    StructLiteralFieldName = 806 => "GUST-E0806";
    /// A `let` pattern that is not one identifier with an optional type.
    LetBindingForm = 807 => "GUST-E0807";
    /// An atomic call whose receiver is not a mutable reference.
    AtomicEmitterReference = 808 => "GUST-E0808";
    /// An atomic call whose receiver is not a buffer element.
    AtomicEmitterElement = 809 => "GUST-E0809";
    /// A struct literal outside a `let` initializer or an assignment.
    StructLiteralPosition = 810 => "GUST-E0810";
    /// An expression form with no lowering yet.
    UnsupportedExpression = 811 => "GUST-E0811";
    /// A literal other than a boolean or a 32-bit number.
    UnsupportedLiteral = 812 => "GUST-E0812";
    /// A type that is not a named path.
    TypeNeedsPath = 813 => "GUST-E0813";
    /// A `Result` without a payload type.
    ResultPayloadMissing = 814 => "GUST-E0814";
    /// A resource whose element type is not a named type.
    ResourceElementPath = 815 => "GUST-E0815";
    /// A resource without angle-bracketed arguments.
    ResourceElementArity = 816 => "GUST-E0816";
    /// A resource without an element type argument.
    ResourceElementMissing = 817 => "GUST-E0817";

    // E09xx: reserved names and shadowing.
    /// An identifier with a reserved `__gust_`, `__gpu_` or `__Gust` prefix.
    ReservedPrefix = 901 => "GUST-E0901";
    /// A local that shadows a resource or dispatch-thread parameter.
    ShadowedParameter = 902 => "GUST-E0902";
}

/// The error for `code` at `span`: the message prefixed with `[GUST-E....]`.
///
/// Message texts are not stable; the code is.
pub(crate) fn error(code: Code, span: impl ToTokens, message: impl Display) -> syn::Error {
    syn::Error::new_spanned(span, format!("[{}] {message}", code.text()))
}

/// [`error`] at a bare span, for a location that has no tokens (the macro call site).
pub(crate) fn error_at(code: Code, span: proc_macro2::Span, message: impl Display) -> syn::Error {
    syn::Error::new(span, format!("[{}] {message}", code.text()))
}

/// [`error`] for a failure inside `parse_nested_meta`, spanned over the option as
/// `ParseNestedMeta::error` is.
pub(crate) fn meta_error(
    meta: &syn::meta::ParseNestedMeta<'_>,
    code: Code,
    message: impl Display,
) -> syn::Error {
    meta.error(format!("[{}] {message}", code.text()))
}

/// A `syn` parse error given `code`, keeping every combined error and its span.
///
/// For failures `syn` reports itself (an unparsable attribute argument list, for
/// example), where there is no message of ours to attach the code to. An error that
/// already carries a code keeps it, so wrapping a result that mixes our errors with
/// `syn`'s is safe.
pub(crate) fn wrap(code: Code, parse_error: syn::Error) -> syn::Error {
    let mut wrapped: Option<syn::Error> = None;
    for part in parse_error {
        let part = if part.to_string().starts_with("[GUST-E") {
            part
        } else {
            syn::Error::new(part.span(), format!("[{}] {part}", code.text()))
        };
        match &mut wrapped {
            Some(existing) => existing.combine(part),
            None => wrapped = Some(part),
        }
    }
    wrapped.expect("a syn::Error holds at least one error")
}

/// `syn::parse2::<T>`, with a failure given `code`.
pub(crate) fn parse2<T: syn::parse::Parse>(
    code: Code,
    tokens: proc_macro2::TokenStream,
) -> syn::Result<T> {
    syn::parse2(tokens).map_err(|error| wrap(code, error))
}

/// `Attribute::parse_args_with`, with a failure given `code`.
pub(crate) fn parse_args_with<P: syn::parse::Parser>(
    attribute: &syn::Attribute,
    code: Code,
    parser: P,
) -> syn::Result<P::Output> {
    attribute
        .parse_args_with(parser)
        .map_err(|error| wrap(code, error))
}

/// `Attribute::parse_nested_meta`, with a failure `syn` words itself given `code`. An
/// error `logic` returns with a code of its own keeps that code (see [`wrap`]).
pub(crate) fn parse_nested_meta(
    attribute: &syn::Attribute,
    code: Code,
    logic: impl FnMut(syn::meta::ParseNestedMeta<'_>) -> syn::Result<()>,
) -> syn::Result<()> {
    attribute
        .parse_nested_meta(logic)
        .map_err(|error| wrap(code, error))
}

/// Helpers that read codes back out of errors, for tests.
#[cfg(test)]
pub(crate) mod testing {
    use std::{
        env,
        fs::OpenOptions,
        io::Write,
        panic::Location,
        sync::{Mutex, PoisonError},
    };

    use super::{ALL, Code};

    /// `syn::parse_str` for tests that build a module from source text.
    pub(crate) fn parse_str<T: syn::parse::Parse>(source: &str) -> syn::Result<T> {
        syn::parse_str(source)
    }

    /// `syn::parse2` for tests that read macro output back.
    pub(crate) fn parse2<T: syn::parse::Parse>(tokens: proc_macro2::TokenStream) -> syn::Result<T> {
        syn::parse2(tokens)
    }

    /// When set, [`assert_code`] and [`assert_codes`] append `<code> <calling file>` to the
    /// file it names; see `every_code_is_asserted_by_a_test_or_listed_unreachable_with_a_reason`.
    pub(crate) const RECORD_VARIABLE: &str = "GUST_RECORD_ASSERTED_CODES";

    /// Whether an assertion made in `file` says something about a rejection. The tests of
    /// this file check the mechanism with codes of their own making, so they do not count.
    pub(crate) fn counts_as_rejection_test(file: &str) -> bool {
        !file
            .replace('\\', "/")
            .ends_with("gust-macros/src/diagnostics.rs")
    }

    fn record(code: Code, caller: &Location<'_>) {
        let Some(path) = env::var_os(RECORD_VARIABLE) else {
            return;
        };
        if !counts_as_rejection_test(caller.file()) {
            return;
        }
        static LOCK: Mutex<()> = Mutex::new(());
        let _held = LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        let line = format!("{} {}\n", code.text(), caller.file().replace('\\', "/"));
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut file| file.write_all(line.as_bytes()))
            .unwrap_or_else(|error| panic!("could not record {}: {error}", code.text()));
    }

    /// The code at the start of `message` (`[GUST-E0401] ...`), or `None` when uncoded.
    pub(crate) fn code_of_message(message: &str) -> Option<Code> {
        let rest = message.strip_prefix('[')?;
        let (text, _) = rest.split_once(']')?;
        ALL.iter().copied().find(|code| code.text() == text)
    }

    /// The code of every error combined in `error`, in order. Panics on an uncoded one,
    /// so a rejection that skipped [`super::error`] fails the test that sees it.
    pub(crate) fn codes_of(error: &syn::Error) -> Vec<Code> {
        error
            .clone()
            .into_iter()
            .map(|part| {
                let message = part.to_string();
                code_of_message(&message)
                    .unwrap_or_else(|| panic!("diagnostic without a code: {message}"))
            })
            .collect()
    }

    /// The codes of a rejection; empty for an accepted input.
    pub(crate) fn codes_of_result<T>(result: &syn::Result<T>) -> Vec<Code> {
        result.as_ref().err().map(codes_of).unwrap_or_default()
    }

    /// Asserts the first diagnostic of `error` has `code`. Like the substring checks it
    /// replaces, this looks at the first of any combined errors; use [`assert_codes`]
    /// to pin all of them.
    ///
    /// This, [`assert_codes`] and [`assert_rendered_code`] are the only ways a test asserts
    /// a rejection's code, and the guard counts a code as tested only when one of them
    /// passed for it.
    #[track_caller]
    pub(crate) fn assert_code(error: &syn::Error, code: Code) {
        assert_code_for(error, code, "");
    }

    /// [`assert_code`] whose failure message also shows `context` (for a table row, the
    /// input that was rejected).
    #[track_caller]
    pub(crate) fn assert_code_for(error: &syn::Error, code: Code, context: &str) {
        let codes = codes_of(error);
        let separator = if context.is_empty() { "" } else { ": " };
        assert_eq!(codes.first(), Some(&code), "{error}{separator}{context}");
        record(code, Location::caller());
    }

    /// Asserts `text`, a rendered diagnostic (the message of a `compile_error!` the macro
    /// returned), carries `[code]`. For output that has already left `syn::Error`; it counts
    /// like [`assert_code`].
    #[track_caller]
    pub(crate) fn assert_rendered_code(text: &str, code: Code) {
        let bracketed = format!("[{}]", code.text());
        assert!(text.contains(&bracketed), "{bracketed} not in: {text}");
        record(code, Location::caller());
    }

    /// Asserts the codes of every combined error, in order.
    #[track_caller]
    pub(crate) fn assert_codes(error: &syn::Error, expected: &[Code]) {
        assert_eq!(codes_of(error), expected, "{error}");
        for &code in expected {
            record(code, Location::caller());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, env, fs, path::Path, process};

    use super::testing::{assert_code, assert_codes, code_of_message, codes_of, codes_of_result};
    use super::*;

    /// Guard (i): numbers are unique, ascending and inside a known family, and the text
    /// is the number. `ALL` is generated with the enum, so no variant can be missing.
    #[test]
    fn codes_are_unique_ordered_and_match_their_text() {
        let mut seen = HashSet::new();
        let mut previous = 0;
        for &code in ALL {
            let number = code as u16;
            assert!(seen.insert(number), "duplicate discriminant {number}");
            assert!(number > previous, "{code:?} is out of order");
            previous = number;
            assert!(
                (1..=9).contains(&(number / 100)),
                "{code:?}: unknown family"
            );
            assert!(
                !number.is_multiple_of(100),
                "{code:?}: number 0 of a family"
            );
            assert_eq!(code.text(), format!("GUST-E{number:04}"), "{code:?}");
        }
        assert_eq!(seen.len(), ALL.len());
    }

    /// The `GUST-E` codes in the headings of `docs/DIAGNOSTICS.md`, outside code fences.
    fn documented_codes() -> Vec<String> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/DIAGNOSTICS.md");
        let text = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("could not read docs/DIAGNOSTICS.md: {error}"));
        let mut in_fence = false;
        let mut codes = Vec::new();
        for line in text.lines() {
            if line.trim_start().starts_with("```") {
                in_fence = !in_fence;
            } else if !in_fence && line.starts_with('#') {
                codes.extend(heading_codes(line));
            }
        }
        codes
    }

    /// Every `GUST-E` plus digits token in a heading line.
    fn heading_codes(line: &str) -> Vec<String> {
        let mut found = Vec::new();
        let mut rest = line;
        while let Some(start) = rest.find("GUST-E") {
            let tail = &rest[start..];
            let end = tail["GUST-E".len()..]
                .find(|c: char| !c.is_ascii_digit())
                .map_or(tail.len(), |digits| "GUST-E".len() + digits);
            found.push(tail[..end].to_owned());
            rest = &tail[end..];
        }
        found
    }

    /// Guard (ii): the doc page and `ALL` list exactly the same codes, each once.
    #[test]
    fn every_code_has_one_documentation_heading_and_no_heading_is_unknown() {
        let documented = documented_codes();
        let unique: HashSet<&String> = documented.iter().collect();
        assert_eq!(
            unique.len(),
            documented.len(),
            "a code has more than one heading: {documented:?}"
        );
        for code in ALL {
            assert!(
                unique.contains(&code.text().to_owned()),
                "{} has no heading in docs/DIAGNOSTICS.md",
                code.text()
            );
        }
        let known: HashSet<&str> = ALL.iter().map(|code| code.text()).collect();
        for heading in &documented {
            assert!(
                known.contains(heading.as_str()),
                "docs/DIAGNOSTICS.md documents {heading}, which is not a code"
            );
        }
    }

    #[test]
    fn heading_codes_reads_only_whole_code_tokens() {
        assert_eq!(heading_codes("### GUST-E0401 loop"), ["GUST-E0401"]);
        assert_eq!(heading_codes("## E04xx control flow"), Vec::<String>::new());
        assert_eq!(
            heading_codes("### GUST-E0401 and GUST-E0402"),
            ["GUST-E0401", "GUST-E0402"]
        );
        assert_eq!(heading_codes("### GUST-E"), ["GUST-E"]);
    }

    #[test]
    fn error_puts_the_code_first_and_keeps_the_message() {
        let error = error(Code::LoopNotSupported, quote::quote!(loop {}), "no loops");
        assert_eq!(error.to_string(), "[GUST-E0401] no loops");
        assert_code(&error, Code::LoopNotSupported);
        assert_codes(&error, &[Code::LoopNotSupported]);
    }

    #[test]
    fn combined_errors_report_their_codes_in_order() {
        let mut error = super::error(Code::WhileNotSupported, quote::quote!(a), "first");
        error.combine(super::error(
            Code::LoopNotSupported,
            quote::quote!(b),
            "second",
        ));
        assert_eq!(
            codes_of(&error),
            [Code::WhileNotSupported, Code::LoopNotSupported]
        );
        assert_code(&error, Code::WhileNotSupported);
    }

    #[test]
    fn an_accepted_input_yields_no_code() {
        assert!(codes_of_result(&Ok(())).is_empty());
        let rejected: syn::Result<()> = Err(super::error(Code::Closure, quote::quote!(a), "x"));
        assert_eq!(codes_of_result(&rejected), [Code::Closure]);
    }

    #[test]
    fn message_without_a_code_is_not_a_code() {
        assert_eq!(code_of_message("plain message"), None);
        assert_eq!(code_of_message("[GUST-E9999] unknown"), None);
        assert_eq!(code_of_message("[GUST-E0401 unclosed"), None);
        assert_eq!(
            code_of_message("[GUST-E0401] known"),
            Some(Code::LoopNotSupported)
        );
    }

    #[test]
    fn wrap_prefixes_every_combined_parse_error_and_keeps_order() {
        let mut parse_error = syn::Error::new(proc_macro2::Span::call_site(), "first problem");
        parse_error.combine(syn::Error::new(
            proc_macro2::Span::call_site(),
            "second problem",
        ));
        let wrapped = wrap(Code::MalformedDerive, parse_error);
        let messages: Vec<String> = wrapped.into_iter().map(|part| part.to_string()).collect();
        assert_eq!(
            messages,
            ["[GUST-E0108] first problem", "[GUST-E0108] second problem"]
        );
    }

    #[test]
    fn wrap_keeps_a_code_an_error_already_has() {
        let mut mixed = error(Code::UnknownKernelOption, quote::quote!(a), "ours");
        mixed.combine(syn::Error::new(proc_macro2::Span::call_site(), "syn's"));
        let wrapped = wrap(Code::MalformedKernelAttribute, mixed);
        assert_codes(
            &wrapped,
            &[Code::UnknownKernelOption, Code::MalformedKernelAttribute],
        );
    }

    // --- Guard (iii): no error is created outside this file ---
    //
    // Clippy does the work: the workspace `clippy.toml` bars `syn::Error::new`, the `syn`
    // parse functions and the rest, and only this file allows them (see the module docs).
    // These tests keep that honest. The config must still list the items, no other file may
    // allow them, and the one form clippy cannot name is checked as text.

    /// The `.rs` files below `dir`, relative to `root` with `/` separators.
    fn rust_files(root: &Path, dir: &Path, found: &mut Vec<String>) {
        let mut entries: Vec<_> = fs::read_dir(dir)
            .unwrap_or_else(|error| panic!("could not read {}: {error}", dir.display()))
            .map(|entry| entry.expect("directory entry").path())
            .collect();
        entries.sort();
        for path in entries {
            if path.is_dir() {
                rust_files(root, &path, found);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                let relative = path.strip_prefix(root).expect("below the root");
                let parts: Vec<_> = relative
                    .components()
                    .map(|part| part.as_os_str().to_string_lossy().into_owned())
                    .collect();
                found.push(parts.join("/"));
            }
        }
    }

    /// The sources of this crate other than this file, as (path below `src`, text).
    fn other_sources() -> Vec<(String, String)> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        rust_files(&root, &root, &mut files);
        for expected in ["lib.rs", "validate/mod.rs", "slang/mod.rs", "expand/mod.rs"] {
            assert!(
                files.iter().any(|file| file == expected),
                "{expected} not scanned: {files:?}"
            );
        }
        files
            .into_iter()
            .filter(|file| file != "diagnostics.rs")
            .map(|file| {
                let source = fs::read_to_string(root.join(&file))
                    .unwrap_or_else(|error| panic!("could not read {file}: {error}"));
                (file, source)
            })
            .collect()
    }

    /// What in a source file can silence the bar: the two lints by name (`allow`, `expect`,
    /// `warn`, `cfg_attr`), the groups that contain them (`clippy::style`, `clippy::all`) and
    /// `warnings`, which an attribute in the source overrides `-D warnings` for.
    const SILENCERS: &[&str] = &[
        "disallowed_",
        "clippy::style",
        "clippy::all",
        "allow(warnings",
        "expect(warnings",
    ];

    fn silencers_in(source: &str) -> Vec<&'static str> {
        SILENCERS
            .iter()
            .copied()
            .filter(|silencer| source.contains(silencer))
            .collect()
    }

    /// A manifest line with whitespace and quotes removed, so `[ workspace . "lints" ]` and
    /// `workspace.lints` compare equal. A key spelled with a TOML unicode escape is not
    /// decoded; deliberate evasion is out of scope (this crate would also have to opt in).
    fn squeezed(text: &str) -> String {
        text.chars()
            .filter(|c| !c.is_whitespace() && *c != '"' && *c != '\'')
            .collect()
    }

    /// The squeezed name of a table header line (`[ "lints" . clippy ] # note` gives
    /// `lints.clippy`), or `None` for any other line.
    fn header_name(line: &str) -> Option<String> {
        let line = line.trim();
        line.starts_with('[')
            .then(|| squeezed(line.trim_start_matches('[').split(']').next().unwrap_or("")))
    }

    /// Whether a manifest has a `[lints]`, `[lints.*]` or `[workspace.lints*]` table, however
    /// the header is spaced or quoted.
    fn has_lints_table(manifest: &str) -> bool {
        manifest
            .lines()
            .filter_map(header_name)
            .any(|name| name.starts_with("lints") || name.starts_with("workspace.lints"))
    }

    /// Whether this crate's own manifest has a lints table in any form: a `[lints` header
    /// (`has_lints_table`) or a dotted or inline key such as `lints.workspace = true`. This
    /// crate must not opt in to workspace lints, which is what keeps them away from it.
    fn crate_manifest_has_lints(manifest: &str) -> bool {
        has_lints_table(manifest)
            || manifest.lines().any(|line| {
                let key = squeezed(line.split('=').next().unwrap_or(""));
                line.contains('=') && (key == "lints" || key.starts_with("lints."))
            })
    }

    /// What the root manifest may not hold in lints: the one allowed table is
    /// `[workspace.lints.rust]` with the single key `unexpected_cfgs` (the check-cfg
    /// declaration of `cfg(gust_dual)`). Every other lints content is named: a package-level
    /// `[lints` table, any other `[workspace.lints` table (a `clippy` group, a key under
    /// `rust.x`), another key in the allowed table, and the dotted or inline forms of
    /// `lints` in `[workspace]` or at the top level. A key spelled with a TOML unicode escape
    /// (a backslash-u escape for one letter of `lints`) is not decoded and gets past this
    /// parser; deliberate evasion is out of scope, since this crate would also have to opt in.
    fn root_lints_offenders(manifest: &str) -> Vec<String> {
        let mut offenders = Vec::new();
        let mut section = String::new();
        for line in manifest.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(name) = header_name(line) {
                if name.starts_with("lints")
                    || (name.starts_with("workspace.lints") && name != "workspace.lints.rust")
                {
                    offenders.push(format!("table [{name}]"));
                }
                section = name;
                continue;
            }
            let Some((key, _)) = line.split_once('=') else {
                continue;
            };
            let key = squeezed(key);
            let top = section.is_empty();
            if section == "workspace.lints.rust" {
                if key != "unexpected_cfgs" {
                    offenders.push(format!("key {key} in [workspace.lints.rust]"));
                }
            } else if (top && (key.starts_with("workspace.lints") || key == "workspace"))
                || key == "lints"
                || key.starts_with("lints.")
                || (section == "workspace" && key.starts_with("lints"))
            {
                offenders.push(format!("key {key} in [{section}]"));
            }
        }
        offenders
    }

    /// Guard (iii), the allowance: nothing but this file may allow the items `clippy.toml`
    /// bars, so an error without a code cannot be created by silencing the lint. Checked:
    /// the source files of this crate (the lints by name, `clippy::style`, `clippy::all`,
    /// `warnings`), a lints table in this crate's manifest (any form), the lints of the root
    /// manifest (only `[workspace.lints.rust] unexpected_cfgs` is allowed; this crate does
    /// not opt in to it), and the config clippy reads. Clippy takes the first `clippy.toml`
    /// or `.clippy.toml` found walking up from the crate's directory, so the root one must
    /// exist and none may sit in
    /// `crates/gust-macros` or `crates`; `CLIPPY_CONF_DIR` must be unset. Not checked:
    /// `rustflags` in a `.cargo/config.toml` (see docs/DIAGNOSTICS.md).
    #[test]
    fn only_this_file_allows_the_barred_syn_items() {
        let mut offenders = Vec::new();
        for (file, source) in other_sources() {
            for silencer in silencers_in(&source) {
                offenders.push(format!("{file}: names {silencer}"));
            }
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let read = |manifest: &str| {
            fs::read_to_string(root.join(manifest))
                .unwrap_or_else(|error| panic!("could not read {manifest}: {error}"))
        };
        if crate_manifest_has_lints(&read("crates/gust-macros/Cargo.toml")) {
            offenders.push("crates/gust-macros/Cargo.toml: has a lints table".to_owned());
        }
        for offender in root_lints_offenders(&read("Cargo.toml")) {
            offenders.push(format!(
                "Cargo.toml: lints beyond unexpected_cfgs: {offender}"
            ));
        }
        assert!(
            root.join("clippy.toml").is_file(),
            "the root clippy.toml is missing"
        );
        for directory in ["crates/gust-macros", "crates"] {
            for name in ["clippy.toml", ".clippy.toml"] {
                if root.join(directory).join(name).exists() {
                    offenders.push(format!("{directory}/{name}: another clippy config"));
                }
            }
        }
        if env::var_os("CLIPPY_CONF_DIR").is_some() {
            offenders.push("CLIPPY_CONF_DIR is set".to_owned());
        }
        assert!(
            offenders.is_empty(),
            "create errors with diagnostics::error and parse through the helpers in diagnostics.rs instead of silencing the bar: {offenders:?}"
        );
    }

    #[test]
    fn silencers_are_found_in_every_form_and_plain_code_has_none() {
        for source in [
            "#[allow(clippy::disallowed_methods)]\nfn f() {}",
            "#![expect(clippy::disallowed_macros)]",
            "#[cfg_attr(test, allow(clippy::disallowed_methods))]",
            "#![allow(clippy::style)]",
            "#[allow(clippy::all)]",
            "#![allow(warnings)]",
            "#![expect(warnings)]",
        ] {
            assert!(!silencers_in(source).is_empty(), "{source}");
        }
        assert!(silencers_in("#[allow(dead_code)]\nfn f() { syn::Ident::new }").is_empty());
    }

    #[test]
    fn a_lints_table_is_a_header_not_a_mention() {
        for manifest in [
            "[lints]\nworkspace = true",
            "[package]\n[lints.clippy]\nstyle = \"allow\"",
            "[workspace.lints.clippy]\n",
            "  [workspace.lints]",
        ] {
            assert!(has_lints_table(manifest), "{manifest}");
        }
        assert!(!has_lints_table(
            "# no lints table here\n[package]\nname = \"x\""
        ));
    }

    const ROOT_OK: &str = "[workspace]\nmembers = [\"x\"]\n\n# why\n[workspace.lints.rust]\nunexpected_cfgs = { level = \"warn\", check-cfg = ['cfg(gust_dual)'] }\n";

    #[test]
    fn the_root_may_declare_exactly_the_unexpected_cfgs_check_cfg() {
        assert_eq!(root_lints_offenders(ROOT_OK), Vec::<String>::new());
        assert_eq!(
            root_lints_offenders("[workspace]\nmembers = []\n"),
            Vec::<String>::new()
        );
        // Spacing and quoting of the header do not change what it is.
        assert!(
            root_lints_offenders("[ workspace . lints . \"rust\" ]\nunexpected_cfgs = {}\n")
                .is_empty()
        );
    }

    #[test]
    fn any_other_root_lints_content_is_an_offender() {
        for (name, extra) in [
            (
                "a clippy group",
                "\n[workspace.lints.clippy]\nstyle = \"warn\"\n",
            ),
            (
                "a clippy lint set to allow",
                "\n[workspace.lints.clippy]\ndisallowed_methods = \"allow\"\n",
            ),
            (
                "the bar lint under rust",
                "disallowed_methods = \"allow\"\n",
            ),
            ("a second rust key", "warnings = \"allow\"\n"),
            ("a dotted rust key", "unexpected_cfgs.level = \"allow\"\n"),
            ("a package-level table", "\n[lints]\nworkspace = true\n"),
            (
                "a package-level group",
                "\n[lints.clippy]\nstyle = \"allow\"\n",
            ),
            (
                "a rust sub-table",
                "\n[workspace.lints.rust.unexpected_cfgs]\nlevel = \"allow\"\n",
            ),
            (
                "a bare lints table",
                "\n[workspace.lints]\nrust = { warnings = \"allow\" }\n",
            ),
            (
                "a spaced header",
                "\n[ workspace . lints . clippy ]\nx = \"allow\"\n",
            ),
            (
                "a quoted header",
                "\n[workspace.\"lints\".clippy]\nx = \"allow\"\n",
            ),
        ] {
            let root = format!("{ROOT_OK}{extra}");
            assert!(!root_lints_offenders(&root).is_empty(), "{name}: {root}");
        }
        // Dotted and inline forms: inside [workspace] and at the top level.
        for root in [
            "[workspace]\nlints.clippy.disallowed_methods = \"allow\"\n",
            "[workspace]\nlints = { clippy = { x = \"allow\" } }\n",
            "workspace.lints.clippy.x = \"allow\"\n[workspace]\n",
            "workspace = { lints = { clippy = { x = \"allow\" } } }\n",
            "lints.workspace = true\n[workspace]\n",
        ] {
            assert!(!root_lints_offenders(root).is_empty(), "{root}");
        }
        // The allowed key in another table is not covered by the allowance.
        assert!(
            !root_lints_offenders("[workspace.lints.clippy]\nunexpected_cfgs = {}\n").is_empty()
        );
    }

    #[test]
    fn this_crates_manifest_may_have_no_lints_in_any_form() {
        for manifest in [
            "[package]\nname = \"x\"\n\n[lints]\nworkspace = true\n",
            "lints.workspace = true\n[package]\nname = \"x\"\n",
            "lints = { workspace = true }\n[package]\n",
            "[lints.clippy]\nstyle = \"allow\"\n",
            "[workspace.lints.rust]\nunexpected_cfgs = {}\n",
            // Spaced, quoted and commented headers are the same tables.
            "[ lints ]\nworkspace = true\n",
            "[\"lints\"]\nworkspace = true\n",
            "['lints'.clippy]\ndisallowed_methods = \"allow\"\n",
            "[ lints . clippy ] # allow it\nx = \"allow\"\n",
        ] {
            assert!(crate_manifest_has_lints(manifest), "{manifest}");
        }
        assert!(!crate_manifest_has_lints(
            "[package]\nname = \"x\"\n[lib]\nproc-macro = true\n"
        ));
    }

    /// The `syn` items the workspace `clippy.toml` must bar (`disallowed-methods`).
    const BARRED_METHODS: &[&str] = &[
        "syn::Error::new",
        "syn::Error::new_spanned",
        "syn::parse::ParseBuffer::error",
        "syn::parse::StepCursor::error",
        "syn::meta::ParseNestedMeta::error",
        "syn::parse::Lookahead1::error",
        "syn::parse",
        "syn::parse2",
        "syn::parse_str",
        "syn::parse_file",
        "syn::parse::Parser::parse",
        "syn::parse::Parser::parse2",
        "syn::parse::Parser::parse_str",
        "syn::Attribute::parse_args",
        "syn::Attribute::parse_args_with",
        "syn::Attribute::parse_nested_meta",
        "syn::MetaList::parse_args",
        "syn::MetaList::parse_args_with",
        "syn::MetaList::parse_nested_meta",
    ];

    /// Guard (iii), the bar: deleting an entry from `clippy.toml` would silently open the
    /// crate to uncoded errors, because clippy has nothing to say about a missing entry.
    #[test]
    fn clippy_toml_bars_the_syn_items_and_names_this_file() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../clippy.toml");
        let text = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("could not read clippy.toml: {error}"));
        let entries: Vec<&str> = text
            .lines()
            .filter(|line| line.trim_start().starts_with("{ path = "))
            .collect();
        let mut problems = Vec::new();
        for item in BARRED_METHODS.iter().chain(&["syn::parse_macro_input"]) {
            let quoted = format!("path = \"{item}\"");
            match entries.iter().find(|entry| entry.contains(&quoted)) {
                None => problems.push(format!("clippy.toml does not bar {item}")),
                Some(entry) if !entry.contains("diagnostics.rs") => {
                    problems.push(format!(
                        "the reason for {item} does not name diagnostics.rs"
                    ));
                }
                Some(_) => {}
            }
        }
        assert!(problems.is_empty(), "{problems:#?}");
    }

    /// Guard (iii), the form clippy cannot name: `syn::Error: From<LexError>` is a trait
    /// impl, which `disallowed-methods` cannot bar for one type. Plain text, so a mention in
    /// a comment fails too; it does not see a `?` on a `LexError` the source never names.
    #[test]
    fn no_other_file_converts_into_a_syn_error() {
        let mut offenders = Vec::new();
        for (file, source) in other_sources() {
            for text in ["LexError", "Error::from("] {
                if source.contains(text) {
                    offenders.push(format!("{file}: {text}"));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "create errors with diagnostics::error so they carry a code: {offenders:?}"
        );
    }

    // --- Reachability: which codes a `#[gpu]` input can produce ---

    /// Codes no `#[gpu]` input can produce, each with the reason. The emitter and the
    /// expander keep these checks as a second line of defence: the validator rejects the
    /// same shape first, so they fire only when a module skips validation. Each has a unit
    /// test that calls the emitter or expander directly. `docs/DIAGNOSTICS.md` marks the
    /// same codes "Unreachable through `#[gpu]`"; a test compares the two.
    const UNREACHABLE: &[(Code, &str)] = &[
        (
            Code::PodStructNamedFields,
            "the validator rejects a struct without named fields (E0104) before the host side is generated",
        ),
        (
            Code::EntryPointDispatchParameter,
            "the validator requires exactly one dispatch-thread parameter on a kernel (E0109)",
        ),
        (
            Code::LetBindingForm,
            "the validator applies the same pattern check to every `let` (E0319)",
        ),
        (
            Code::AtomicEmitterReference,
            "the validator checks the first argument of every atomic call (E0602, E0608)",
        ),
        (
            Code::AtomicEmitterElement,
            "the validator checks the first argument of every atomic call (E0609)",
        ),
        (
            Code::ResultPayloadMissing,
            "the validator accepts `Result` only as `Result<T, T>` (E0504)",
        ),
        (
            Code::ResourceElementPath,
            "the emitter rejects a resource type that is not a path as E0813 first",
        ),
        (
            Code::ProfileRefusal,
            "the validator runs under the module's grid profile, which refuses no construct, and a pixel pin is checked by the classifier first (E0723); a test profile raises it",
        ),
    ];

    fn is_listed_unreachable(code: Code) -> bool {
        UNREACHABLE.iter().any(|(listed, _)| *listed == code)
    }

    #[test]
    fn the_unreachable_list_names_each_code_once_with_a_reason() {
        let mut seen = HashSet::new();
        for (code, reason) in UNREACHABLE {
            assert!(seen.insert(*code as u16), "{code:?} is listed twice");
            assert!(ALL.contains(code), "{code:?} is not a code");
            assert!(reason.len() > 20, "{code:?} needs a reason");
        }
    }

    /// Whether an assertion in `file` exercises the validator and emitter together as `#[gpu]`
    /// does (the regression tests, the map tests, and the validator's own tests).
    fn is_pipeline_test_file(file: &str) -> bool {
        let file = file.replace('\\', "/");
        file.ends_with("src/tests/regression.rs")
            || file.ends_with("src/tests/map.rs")
            || file.ends_with("src/tests/workgroup.rs")
            || file.ends_with("src/tests/grid.rs")
            || file.ends_with("src/tests/par_for.rs")
            || file.ends_with("src/tests/manifest.rs")
            || file.ends_with("src/tests/variant.rs")
            || file.ends_with("src/tests/subgroup.rs")
            || file.ends_with("src/validate/mod.rs")
            || file.ends_with("src/validate/limits.rs")
            || file.ends_with("src/validate/classify/tests.rs")
    }

    /// The problems with `recorded`, the (code text, calling file) pairs that
    /// `testing::assert_code` and `testing::assert_codes` wrote after an assertion passed.
    /// A reachable code needs one from a pipeline test file; a listed code needs one from
    /// any test file, so it cannot rot unseen. Naming `Code::X` in a test records nothing.
    fn unasserted(recorded: &[(String, String)]) -> Vec<String> {
        let files_for = |code: Code| {
            recorded
                .iter()
                .filter(move |(text, _)| text == code.text())
                .map(|(_, file)| file.as_str())
        };
        let mut problems = Vec::new();
        for &code in ALL {
            if is_listed_unreachable(code) {
                if files_for(code).next().is_none() {
                    problems.push(format!(
                        "{code:?} is listed UNREACHABLE but no test asserts it"
                    ));
                }
            } else if !files_for(code).any(is_pipeline_test_file) {
                problems.push(format!(
                    "{code:?} is asserted by no test of tests/regression.rs or the validator tests: add an assert_rejected row, or list it in UNREACHABLE with a reason"
                ));
            }
        }
        problems
    }

    /// Every code is asserted by a test that runs the validator or the validator and emitter
    /// together as `#[gpu]` does (`tests/regression.rs` and the validator's own tests), or is
    /// listed in `UNREACHABLE`; and a listed code is asserted directly by some test.
    ///
    /// "Asserted" is measured, not read from the source: the test binary runs again with
    /// every other test (this one skipped) and `testing::assert_code` and `assert_codes`
    /// append the code and the calling file to a file when an assertion passes. Rows of a
    /// table that feed `assert_rejected` therefore count, and a `Code::X` that is only named
    /// (in a comment, a `let`, an unused table) does not.
    #[test]
    fn every_code_is_asserted_by_a_test_or_listed_unreachable_with_a_reason() {
        if env::var_os(testing::RECORD_VARIABLE).is_some() {
            return; // The recording run itself.
        }
        let record = env::temp_dir().join(format!("gust-asserted-codes-{}.txt", process::id()));
        let _ = fs::remove_file(&record);
        let output = process::Command::new(env::current_exe().expect("the test binary"))
            .env(testing::RECORD_VARIABLE, &record)
            // A bless run must not make the recording run rewrite fixtures as well.
            .env_remove("BLESS_GOLDEN")
            .args(["--skip", "every_code_is_asserted_by_a_test"])
            .output()
            .expect("the recording run starts");
        let text = fs::read_to_string(&record).unwrap_or_default();
        let _ = fs::remove_file(&record);
        let recorded: Vec<(String, String)> = text
            .lines()
            .filter_map(|line| line.split_once(' '))
            .map(|(code, file)| (code.to_owned(), file.to_owned()))
            .collect();
        let mut problems = unasserted(&recorded);
        if !output.status.success() {
            let log = String::from_utf8_lossy(&output.stdout);
            let tail: Vec<&str> = log.lines().rev().take(40).collect();
            problems.push(format!(
                "the recording run failed, so some assertions never ran:\n{}",
                tail.into_iter().rev().collect::<Vec<_>>().join("\n")
            ));
        }
        assert!(problems.is_empty(), "{problems:#?}");
    }

    #[test]
    fn a_code_counts_only_when_an_assertion_passed_in_a_file_that_counts() {
        let regression = "crates/gust-macros/src/tests/regression.rs";
        let emitter = "crates/gust-macros/src/slang/mod.rs";
        let all: Vec<(String, String)> = ALL
            .iter()
            .map(|code| (code.text().to_owned(), regression.to_owned()))
            .collect();
        assert_eq!(unasserted(&all), Vec::<String>::new());
        // Nothing recorded, which is what a code that is only named in a test leaves.
        assert_eq!(unasserted(&[]).len(), ALL.len());
        // One code missing.
        let without: Vec<_> = all
            .iter()
            .filter(|(text, _)| text != Code::WhileNotSupported.text())
            .cloned()
            .collect();
        let problems = unasserted(&without);
        assert_eq!(problems.len(), 1, "{problems:#?}");
        assert!(
            problems[0].starts_with("WhileNotSupported "),
            "{problems:#?}"
        );
        // A reachable code asserted only below the pipeline is not enough ...
        let mut moved = all.clone();
        for entry in &mut moved {
            if entry.0 == Code::LoopNotSupported.text() {
                entry.1 = emitter.to_owned();
            }
        }
        let problems = unasserted(&moved);
        assert_eq!(problems.len(), 1, "{problems:#?}");
        assert!(
            problems[0].starts_with("LoopNotSupported "),
            "{problems:#?}"
        );
        // ... but a listed one only needs an assertion somewhere, and Windows paths work.
        let mut listed = all.clone();
        for entry in &mut listed {
            if entry.0 == Code::LetBindingForm.text() {
                entry.1 = emitter.replace('/', "\\");
            }
        }
        assert_eq!(unasserted(&listed), Vec::<String>::new());
        let mut windows = all;
        for entry in &mut windows {
            entry.1 = entry.1.replace('/', "\\");
        }
        assert_eq!(unasserted(&windows), Vec::<String>::new());
    }

    #[test]
    fn assertions_made_in_this_file_are_not_rejection_tests() {
        assert!(!testing::counts_as_rejection_test(
            "crates/gust-macros/src/diagnostics.rs"
        ));
        assert!(!testing::counts_as_rejection_test(
            r"crates\gust-macros\src\diagnostics.rs"
        ));
        // Only this crate's file: a test file that happens to share the name counts.
        assert!(testing::counts_as_rejection_test(
            "crates/gust-macros/src/tests/diagnostics.rs"
        ));
        assert!(testing::counts_as_rejection_test(
            "crates/gust-macros/src/validate/mod.rs"
        ));
    }

    /// One `### GUST-E....` section of the doc page.
    struct DocSection {
        code: String,
        body: Vec<String>,
    }

    /// The sections of the doc page: each heading with exactly one code starts one, and any
    /// other heading ends it. Fenced lines are never headings.
    fn doc_sections(text: &str) -> Vec<DocSection> {
        let mut sections: Vec<DocSection> = Vec::new();
        let mut open = false;
        let mut in_fence = false;
        for line in text.lines() {
            if line.trim_start().starts_with("```") {
                in_fence = !in_fence;
            }
            if !in_fence && line.starts_with('#') {
                open = false;
                if let [code] = heading_codes(line).as_slice() {
                    sections.push(DocSection {
                        code: code.clone(),
                        body: Vec::new(),
                    });
                    open = true;
                }
            } else if open {
                sections
                    .last_mut()
                    .expect("an open section")
                    .body
                    .push(line.to_owned());
            }
        }
        sections
    }

    const REACHED: &str = "Reached through `#[gpu]`";
    const NOT_REACHED: &str = "Unreachable through `#[gpu]`";

    fn doc_text() -> String {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/DIAGNOSTICS.md");
        fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("could not read docs/DIAGNOSTICS.md: {error}"))
    }

    /// Problems with the reachability statement and the opening line of each section.
    fn section_problems(
        sections: &[DocSection],
        unreachable: &dyn Fn(&str) -> bool,
    ) -> Vec<String> {
        let mut problems = Vec::new();
        for section in sections {
            let body = section.body.join("\n");
            let reached = body.matches(REACHED).count();
            let not_reached = body.matches(NOT_REACHED).count();
            if reached + not_reached != 1 {
                problems.push(format!(
                    "{}: must say exactly once whether it is reached through `#[gpu]` ({reached} + {not_reached})",
                    section.code
                ));
            } else if (not_reached == 1) != unreachable(&section.code) {
                problems.push(format!(
                    "{}: the doc says {} but the UNREACHABLE list says the opposite",
                    section.code,
                    if not_reached == 1 {
                        "unreachable"
                    } else {
                        "reached"
                    }
                ));
            }
            let first = section
                .body
                .iter()
                .map(|line| line.trim())
                .find(|line| !line.is_empty())
                .unwrap_or("");
            if [
                "Rejected:",
                REACHED,
                NOT_REACHED,
                "Write instead:",
                "Decision:",
            ]
            .iter()
            .any(|label| first.starts_with(label))
            {
                problems.push(format!(
                    "{}: needs a one-line meaning before `Rejected:`",
                    section.code
                ));
            }
        }
        problems
    }

    /// Each section of the doc page says whether `#[gpu]` can reach its code, matching
    /// `UNREACHABLE`, and opens with a line saying what the code means.
    #[test]
    fn the_doc_marks_reachability_as_the_unreachable_list_does_and_opens_with_a_meaning() {
        let sections = doc_sections(&doc_text());
        assert_eq!(sections.len(), ALL.len(), "one section per code");
        let unreachable: HashSet<&str> = UNREACHABLE.iter().map(|(code, _)| code.text()).collect();
        let problems = section_problems(&sections, &|code| unreachable.contains(code));
        assert!(problems.is_empty(), "{problems:#?}");
    }

    #[test]
    fn section_problems_finds_missing_conflicting_and_misplaced_statements() {
        let sample = "## E01xx\n\
            ### GUST-E0101 ok\n\nMeans a thing.\nReached through `#[gpu]`.\nRejected: x.\n\n\
            ### GUST-E0102 unreachable ok\n\nMeans a thing.\nUnreachable through `#[gpu]`: defence in depth.\nRejected: x.\n\n\
            ### GUST-E0103 no statement\n\nMeans a thing.\nRejected: x.\n\n\
            ### GUST-E0104 both\n\nMeans.\nReached through `#[gpu]`. Unreachable through `#[gpu]`.\n\n\
            ### GUST-E0105 wrong\n\nMeans.\nReached through `#[gpu]`.\n\n\
            ### GUST-E0106 no meaning\n\nReached through `#[gpu]`.\nRejected: x.\n\n\
            ### GUST-E0107 fenced\n\n```text\n### GUST-E0199 not a heading\n```\nMeans.\nReached through `#[gpu]`.\n\
            ## Next family\nReached through `#[gpu]`.\n";
        let sections = doc_sections(sample);
        let codes: Vec<&str> = sections.iter().map(|s| s.code.as_str()).collect();
        assert_eq!(
            codes,
            [
                "GUST-E0101",
                "GUST-E0102",
                "GUST-E0103",
                "GUST-E0104",
                "GUST-E0105",
                "GUST-E0106",
                "GUST-E0107"
            ]
        );
        let problems = section_problems(&sections, &|code| {
            matches!(code, "GUST-E0102" | "GUST-E0105")
        });
        assert_eq!(problems.len(), 4, "{problems:#?}");
        assert!(problems[0].starts_with("GUST-E0103: must say exactly once"));
        assert!(problems[1].starts_with("GUST-E0104: must say exactly once"));
        assert!(problems[2].starts_with("GUST-E0105: the doc says reached"));
        assert!(problems[3].starts_with("GUST-E0106: needs a one-line meaning"));
    }
}
