//! The stage profile (decision D63, slice mx.p0): every rule that depends on the execution shape of a kernel, read
//! through one interface instead of being spelled at its call site.
//!
//! There are two shapes. `grid` (a compute dispatch over a one-, two- or three-dimensional thread grid) is the
//! default, and [`GRID`] is its profile: the text and the rules that the emitter, the validator and the twin harness
//! used to hold inline, moved here without changing a byte of output. `pixel` (a draw whose invocations are the
//! fragments of a full-screen pass, slice mx.p1a) is [`PIXEL`]; a kernel pinned `#[kernel(shape = pixel)]` is built
//! with it and no call site changes.
//!
//! The trait receives facts and answers with text, a table or a verdict. It never walks the syn body of a kernel:
//! the validator and the emitter keep their visitors, and a hook that needs the body is a sign that the shape does
//! not fit the interface (the kill rule of plan section 2.4, which this module is checked against).

use std::fmt::Write as _;

use crate::slang::ParameterFlavor;

mod grid;
mod pixel;

pub(crate) use grid::GRID;
pub(crate) use pixel::PIXEL;

/// The execution shapes; the type exists so that a profile can name itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Shape {
    /// One compute thread per element of a dispatch grid (`[shader("compute")]`, `SV_DispatchThreadID`).
    Grid,
    /// One fragment per element of a full-screen pass (`[shader("fragment")]`, `SV_Position`); every output stays a
    /// storage buffer, written at the fragment's own index (D120).
    Pixel,
}

/// What an invocation parameter of a kernel stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Builtin {
    /// The global id of the invocation (`SV_DispatchThreadID`).
    Global,
    /// The id within the workgroup, derived from the global id and the workgroup size (`LocalId`).
    Local,
    /// The id of the workgroup, derived like [`Builtin::Local`] (`GroupId`).
    Group,
}

/// Which kinds of parameter a shape admits and how they are numbered.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BindingRules {
    /// The refusal text for a plain value parameter of a kernel (GUST-E0801).
    pub(crate) value_refusal: &'static str,
    /// The refusal text for a kernel without the dispatch-thread parameter, from the emitter.
    pub(crate) missing_dispatch_emit: &'static str,
    /// The refusal text for a kernel without exactly one dispatch-thread parameter, from the validator.
    pub(crate) missing_dispatch_validate: &'static str,
    /// The parameter flavors that take a binding slot in descriptor set 0, numbered in declaration order.
    slotted: &'static [ParameterFlavor],
    /// The typed call has the `call_extent`, `submit_extent` and `dispatch_extent` forms over a `gust::Extent`. A shape
    /// whose invocations are not a grid has the one-dimensional forms only.
    pub(crate) extent_calls: bool,
}

impl BindingRules {
    pub(crate) const fn new(
        value_refusal: &'static str,
        missing_dispatch_emit: &'static str,
        missing_dispatch_validate: &'static str,
        slotted: &'static [ParameterFlavor],
    ) -> Self {
        Self {
            value_refusal,
            missing_dispatch_emit,
            missing_dispatch_validate,
            slotted,
            extent_calls: true,
        }
    }

    /// The same rules for a shape whose calls are one-dimensional: no extent forms.
    pub(crate) const fn one_dimensional(mut self) -> Self {
        self.extent_calls = false;
        self
    }

    /// Whether a parameter of this flavor is declared as a resource with a binding slot.
    pub(crate) fn gets_slot(&self, flavor: ParameterFlavor) -> bool {
        self.slotted.contains(&flavor)
    }
}

/// The options of `#[kernel(..)]` that a shape accepts.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OptionRules {
    /// `workgroup_size(x, y, z)` is accepted.
    pub(crate) workgroup_size: bool,
    /// `cpu = "unsupported"` is accepted.
    pub(crate) cpu: bool,
    /// The workgroup size of a kernel that does not give one.
    pub(crate) default_workgroup: [u32; 3],
    /// The text of the error for any other option.
    pub(crate) expected: &'static str,
}

/// What a kernel declares that a shape may refuse, and the findings a profile hands back. The validator fills in the
/// facts from the signature and its own visitors; the profile only judges them.
#[derive(Clone, Debug, Default)]
pub(crate) struct BodyChecks {
    /// The kernel declares `Workgroup<[T; N]>` memory.
    pub(crate) workgroup_memory: bool,
    /// The kernel calls a workgroup barrier.
    pub(crate) barrier: bool,
    /// The kernel uses atomic operations.
    pub(crate) atomics: bool,
    /// Constructs the shape forbids, each as the sentence the diagnostic carries. Empty for every `grid` kernel.
    pub(crate) forbidden: Vec<String>,
}

/// What the emitter knows about a kernel when it asks a profile for text. Built once per kernel from the parameter
/// list, so a profile never sees the syn tree.
#[derive(Clone, Copy, Debug)]
pub(crate) struct KernelFacts<'a> {
    /// The entry point's name (`gpu_<module>_<kernel>`).
    pub(crate) entry_point: &'a str,
    /// The workgroup size.
    pub(crate) workgroup_size: [u32; 3],
    /// The name of the invocation parameter.
    pub(crate) invocation: &'a str,
    /// The storage buffers, in binding order: the prologue reads the length of each.
    pub(crate) resources: &'a [String],
    /// The ids derived from the invocation, in parameter order.
    pub(crate) derived_ids: &'a [DerivedId],
    /// The next free binding slot in descriptor set 0, after the kernel's own resources.
    pub(crate) next_binding: u32,
}

/// An id derived from the global id and the workgroup size: a local id or a group id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DerivedId {
    pub(crate) name: String,
    /// `true` for a group id (`id / size`), `false` for a local id (`id % size`).
    pub(crate) group: bool,
}

/// A resource the shape declares beside the kernel's own, taking the next binding slot. The emitter numbers it. No
/// profile returns one: `pixel` bakes its row width into the shader instead of reading it from a uniform (mx.p1a,
/// option S), because a runtime-owned binding has no place in the descriptor and the typed call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExtraBinding {
    pub(crate) name: String,
    /// The element type of the `ConstantBuffer` it is declared as.
    pub(crate) slang_type: String,
}

/// A write the kernel makes to an output buffer at its own index (`out[id] = value`), as a profile that has to turn
/// it into something else sees it.
#[derive(Clone, Copy, Debug)]
#[allow(
    dead_code,
    reason = "read by the profile of a later shape; `grid` leaves the write as it is"
)]
pub(crate) struct OwnIndexWrite<'a> {
    pub(crate) buffer: &'a str,
    pub(crate) value: &'a str,
}

/// How the CPU twin visits the invocations of a call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TwinHarness {
    /// In order of the linear global id over the call's extent, with the workgroup size deriving the local and the
    /// group ids.
    Linear,
}

/// What the reflection of the compiled artifact must show for a kernel of the shape (D17).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReflectionKind {
    /// A compute entry point with the kernel's workgroup size.
    Compute,
    /// A fragment entry point: the document has a stage and resources and no workgroup size. The descriptor and the
    /// record carry `[1, 1, 1]` (a workgroup size of 0 is refused by the twin and the runtime).
    Fragment,
}

/// The `GetDimensions` reads of the buffers' lengths and strides that open a kernel's body: `__gpu_len_<name>` and
/// `__gpu_stride_<name>`, four spaces in, each line ending in a line feed. Every shape starts its body with them.
pub(crate) fn length_reads(resources: &[String]) -> String {
    let mut text = String::new();
    for resource in resources {
        writeln!(text, "    uint __gpu_len_{resource};").unwrap();
        writeln!(text, "    uint __gpu_stride_{resource};").unwrap();
        writeln!(
            text,
            "    {resource}.GetDimensions(__gpu_len_{resource}, __gpu_stride_{resource});"
        )
        .unwrap();
    }
    text
}

/// How much of the kernel's contract the shape guarantees to the host (D62).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ContractTier {
    /// Every element of every output is the value the Rust body computes.
    Exact,
}

/// The rules of one execution shape. Fourteen methods: each is a table or a short function over facts.
pub(crate) trait StageProfile {
    /// The shape this profile implements: the expansion writes it into the descriptor and, for a shape other than
    /// `grid`, into the export record (mx.k).
    fn shape(&self) -> Shape;

    /// The Slang stage of the entry point: the argument of `[shader("..")]`.
    fn slang_stage(&self) -> &'static str;

    /// Whether a parameter type is an invocation built-in of this shape and which one.
    fn invocation(&self, ty: &syn::Type) -> Option<Builtin>;

    /// The parameter kinds the shape admits and which of them take a binding slot.
    fn binding_rules(&self) -> BindingRules;

    /// The `#[kernel(..)]` options the shape accepts.
    fn options(&self) -> OptionRules;

    /// Judges the constructs a kernel declares; pushes a sentence to `cx.forbidden` for each the shape refuses.
    fn validate_body(&self, cx: &mut BodyChecks);

    /// The attribute lines and the signature of the entry point, without the `[shader(..)]` line and without the
    /// opening brace.
    fn entry_signature(&self, k: &KernelFacts<'_>) -> String;

    /// The statements before the kernel's own body: the length reads of the buffers and the derived ids. Each line
    /// ends in a line feed.
    fn prologue(&self, k: &KernelFacts<'_>) -> String;

    /// The statements after the kernel's own body. Empty for a shape whose entry point returns nothing.
    fn epilogue(&self, k: &KernelFacts<'_>) -> String;

    /// The text of an own-index output write, or `None` for the plain index assignment.
    #[allow(
        dead_code,
        reason = "called by the write classifier of mx.c; until then no write is marked"
    )]
    fn lower_output_write(&self, write: &OwnIndexWrite<'_>) -> Option<String>;

    /// Resources the shape adds to the kernel's own.
    fn extra_bindings(&self, k: &KernelFacts<'_>) -> Vec<ExtraBinding>;

    /// How the CPU twin iterates.
    fn twin(&self) -> TwinHarness;

    /// What the reflection of the artifact must show.
    fn reflection(&self) -> ReflectionKind;

    /// The contract the shape gives the host.
    #[allow(
        dead_code,
        reason = "read by the record writer of mx.k; D62 reserves the second tier"
    )]
    fn contract_tier(&self) -> ContractTier;
}
