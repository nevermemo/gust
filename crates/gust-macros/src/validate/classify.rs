//! The shape classifier (decision D63, plan section 5, slice mx.c): which execution shapes a kernel can take, and for
//! the shapes it cannot, the construct that rules each out.
//!
//! The rules are syntactic and conservative (plan 5.1). Nothing is inferred through aliases, arithmetic or data flow:
//! whatever a rule does not recognise makes the kernel `grid`-only, with the construct named. The classifier never
//! reads the `own_index` stamp of the map generator (m7.c's), and the generator's own kernel and the `par_for!`
//! kernels classify as `grid`-only by these rules (a loop variable, a guard on a uniform, an explicit workgroup size).
//!
//! A kernel is `pixel`-compatible when all of these hold:
//!
//! - The invocation identity is immutable: the parameter is not `mut`, and nothing assigns to it or to a field of it.
//! - The own index is one spelling: `id.x`, `id.global_id().x`, or exactly one immutable `let i = <that>;` whose name
//!   is bound nowhere else in the body (not by a second `let`, a loop variable or a pattern).
//! - Each output (a `StorageMut` parameter that is written) is used only as `out[own] = v` and `out.len()`: no read,
//!   no compound assignment, no component write, no rebinding, no helper argument, no write at another index. A
//!   `StorageMut` that is never written is an input read through a mutable binding.
//! - Every output is definitely written, and the body has no `return`. Definite assignment is ordinary: a block
//!   writes if some statement does, `if c { A } else { B }` if both arms do, `match` if every arm does, a loop never,
//!   and `if c { A }` without `else` only when `c` is transparent: every conjunct is `own < o.len()` (or
//!   `o.len() > own`) for an output `o`, which under the `Shared`-only length rule is always true for a live pixel.
//! - Nothing a fragment cannot carry or do: no workgroup memory, barrier, `LocalId`/`GroupId` or other invocation
//!   built-in than the global id, no atomic, no storage texture, no workgroup size, at least one output, every
//!   output a `u32`, `i32`, `f32`, or a 2- or 4-wide vector of them, at most 8 outputs and 32 bytes of them (D62).
//!
//! The result is the descriptor's `compatible_shapes` (nothing at run time reads it) and the check of the pin
//! `#[kernel(shape = ..)]` (plan 5.2 and 5.3).

use std::collections::HashSet;

use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::ext::IdentExt as _;
use syn::visit::{self, Visit};

use super::barrier::is_barrier_call;
use super::{atomic_buffer_names, atomic_operation_name};
use crate::diagnostics::{self, Code};
use crate::profile::StageProfile;
use crate::slang::{ParameterFlavor, local_binding, parameter_flavor_in, wrapper_element_type};

/// The most colour targets a `pixel` kernel may write (D62).
const MAX_TARGETS: usize = 8;

/// The most bytes of colour targets a `pixel` kernel may write together (D62).
const MAX_TARGET_BYTES: u32 = 32;

/// An execution shape a kernel can take. `cpu` is a target (D51), never a shape.
///
/// This is the classifier's own set; mx.p1a unifies it with the profile's `Shape` when the `pixel` profile exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Shape {
    /// One compute thread per element of a dispatch grid.
    Grid,
    /// One fragment invocation per element of a full-screen pass.
    Pixel,
}

impl Shape {
    /// The classifier's name for a shape of the stage profile.
    pub(crate) fn from_profile(shape: crate::profile::Shape) -> Self {
        match shape {
            crate::profile::Shape::Grid => Self::Grid,
        }
    }

    /// The name in the pin and in the descriptor: `grid` or `pixel`.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Grid => "grid",
            Self::Pixel => "pixel",
        }
    }

    /// The path of `gust::Shape` for this shape, as the descriptor spells it.
    pub(crate) fn tokens(self) -> TokenStream {
        match self {
            Self::Grid => quote!(::gust::Shape::Grid),
            Self::Pixel => quote!(::gust::Shape::Pixel),
        }
    }
}

/// What the `#[kernel(..)]` attribute says that the classifier needs: the shape pin, and the workgroup size when the
/// kernel sets one (the parser fills in the default otherwise, which is not "set").
#[derive(Clone, Debug, Default)]
pub(crate) struct AttributeFacts {
    /// `shape = ..`, at most once.
    pub(crate) shape: Option<ShapePin>,
    /// The tokens of an explicit `workgroup_size(x, y, z)`.
    pub(crate) workgroup_size: Option<TokenStream>,
}

/// A parsed `shape = <identifier>` option.
#[derive(Clone, Debug)]
pub(crate) struct ShapePin {
    pub(crate) shape: Shape,
    /// The tokens of the option (`shape = pixel`), which a diagnostic about the pin is spanned over.
    pub(crate) tokens: TokenStream,
}

/// Why a construct rules a shape out. Each reason has one diagnostic code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Why {
    /// `Workgroup<[T; N]>` memory.
    WorkgroupMemory,
    /// A `workgroup_barrier()` call.
    Barrier,
    /// A `LocalId` or `GroupId` parameter: an invocation built-in other than the global id.
    InvocationBuiltin,
    /// An atomic operation.
    Atomic,
    /// A `StorageTexture2D` or `WriteTexture2D` parameter.
    StorageTexture,
    /// An output whose element type has no colour-target format.
    TargetElement,
    /// More targets than D62 allows.
    TargetCount,
    /// More target bytes than D62 allows.
    TargetBytes,
    /// No output buffer at all.
    NoOutput,
    /// An explicit `workgroup_size`.
    WorkgroupSize,
    /// A `mut` invocation parameter, or an assignment to it.
    MutableInvocation,
    /// A write to an output at an index other than the element's own.
    OwnIndexWrite,
    /// A second alias of the own index, or an alias that is `mut` or bound again.
    OwnIndexAlias,
    /// An output that is not definitely written.
    ConditionalWrite,
    /// A `return` in the kernel body.
    Return,
    /// A read of an output element.
    OutputRead,
    /// A compound assignment to an output element.
    OutputCompound,
    /// A write to one component of an output element.
    OutputComponent,
    /// Any other mention of an output (a rebinding, a helper argument).
    OutputOther,
    /// A `StorageMut` parameter that is never written.
    OutputUnwritten,
}

impl Why {
    /// The diagnostic code of a pin that is refused for this reason.
    pub(crate) fn code(self) -> Code {
        match self {
            Self::MutableInvocation => Code::ShapeMutableInvocation,
            Self::OwnIndexWrite | Self::OwnIndexAlias => Code::ShapeOwnIndex,
            Self::ConditionalWrite => Code::ShapeConditionalWrite,
            Self::Return => Code::ShapeReturn,
            Self::OutputRead
            | Self::OutputCompound
            | Self::OutputComponent
            | Self::OutputOther
            | Self::OutputUnwritten => Code::ShapeOutputUse,
            Self::WorkgroupMemory
            | Self::Barrier
            | Self::InvocationBuiltin
            | Self::Atomic
            | Self::StorageTexture
            | Self::TargetElement
            | Self::TargetCount
            | Self::TargetBytes
            | Self::NoOutput
            | Self::WorkgroupSize => Code::ShapeExcluded,
        }
    }

    /// What the construct is, after its text in the message: the reason in a clause. `detail` is the exclusion's own.
    fn clause(self, detail: &str) -> String {
        match self {
            Self::WorkgroupMemory => {
                "is workgroup memory, which a fragment shader does not have".to_owned()
            }
            Self::Barrier => {
                "is a workgroup barrier, which a fragment shader does not have".to_owned()
            }
            Self::InvocationBuiltin => {
                "is an invocation id other than the global id, and a pixel has only the \
                                        global id"
                    .to_owned()
            }
            Self::Atomic => {
                "is an atomic operation, which the `pixel` shape does not take".to_owned()
            }
            Self::StorageTexture => {
                "is a storage texture, which stays a `grid` resource (D64)".to_owned()
            }
            Self::TargetElement => format!(
                "has {detail}, which is no colour-target format: a `pixel` output is a `u32`, `i32`, `f32` or a \
                 2- or 4-wide vector of them"
            ),
            Self::TargetCount => format!(
                "is output number {detail}, and a `pixel` kernel has at most {MAX_TARGETS} outputs (D62)"
            ),
            Self::TargetBytes => format!(
                "takes the outputs to {detail} bytes, and a `pixel` kernel has at most {MAX_TARGET_BYTES} bytes of \
                 outputs (D62)"
            ),
            Self::NoOutput => "writes no output buffer, so a pixel would carry nothing".to_owned(),
            Self::WorkgroupSize => {
                "sets a workgroup size, which a fragment shader has no use for".to_owned()
            }
            Self::MutableInvocation => format!(
                "{detail} the invocation id, which is fixed for a pixel: declare it without `mut` and never assign \
                 to it"
            ),
            Self::OwnIndexWrite => "writes at an index other than the element's own".to_owned(),
            Self::OwnIndexAlias => format!(
                "{detail}: the own index is `id.x`, `id.global_id().x` or one immutable `let i = id.x;` that is \
                 never bound again"
            ),
            Self::ConditionalWrite => format!(
                "makes the write to the output {detail} conditional, and a pixel output must be written on every \
                 path"
            ),
            Self::Return => "returns early, and a pixel kernel has no `return`".to_owned(),
            Self::OutputRead => {
                format!("reads the output {detail}, which a pixel kernel may only write")
            }
            Self::OutputCompound => format!(
                "is a compound assignment, which reads the element of the output {detail} before writing it"
            ),
            Self::OutputComponent => format!(
                "writes one component of an element of the output {detail}: write the whole element"
            ),
            Self::OutputOther => {
                format!("uses the output {detail} other than as `out[own] = v` and `out.len()`")
            }
            Self::OutputUnwritten => format!(
                "is a mutable buffer that is never written, an input read through a mutable binding; {detail} \
                 should be a plain `Storage` buffer"
            ),
        }
    }
}

/// One construct that rules a shape out.
#[derive(Clone, Debug)]
pub(crate) struct Exclusion {
    /// The shape ruled out (`Pixel`; `grid` takes every kernel the macro accepts).
    pub(crate) shape: Shape,
    pub(crate) why: Why,
    /// The construct as source text.
    pub(crate) construct: String,
    /// Details the reason's clause uses.
    pub(crate) detail: String,
}

impl Exclusion {
    /// The message of a pin refused for this exclusion, without the code.
    pub(crate) fn message(&self, kernel: &str) -> String {
        format!(
            "shape `{}` is not compatible with kernel `{kernel}`: `{}` {} (that makes the kernel `grid`-only)",
            self.shape.name(),
            self.construct,
            self.why.clause(&self.detail),
        )
    }
}

/// The shapes a kernel can take, and what rules out the others.
#[derive(Clone, Debug)]
pub(crate) struct Classification {
    /// Every shape the kernel can take; `Grid` always, then `Pixel` when nothing rules it out.
    pub(crate) shapes: Vec<Shape>,
    /// What rules out `Pixel`, in the order the classifier found it: the attribute, the parameters, then the body.
    pub(crate) exclusions: Vec<Exclusion>,
}

impl Classification {
    /// Whether the kernel can take `shape`.
    pub(crate) fn allows(&self, shape: Shape) -> bool {
        self.shapes.contains(&shape)
    }

    /// The first construct that rules `shape` out.
    pub(crate) fn first_exclusion(&self, shape: Shape) -> Option<&Exclusion> {
        self.exclusions
            .iter()
            .find(|exclusion| exclusion.shape == shape)
    }
}

/// Classifies `function`, a kernel the validator accepted, for the profile `stage` and the attribute `attributes`.
pub(crate) fn classify(
    stage: &dyn StageProfile,
    function: &syn::ItemFn,
    attributes: &AttributeFacts,
) -> Classification {
    let mut found = Found::default();

    if let Some(tokens) = &attributes.workgroup_size {
        found.exclude(Why::WorkgroupSize, tokens, "");
    }

    // The parameters.
    let mut invocation: Option<String> = None;
    let mut outputs: Vec<Output<'_>> = Vec::new();
    for argument in &function.sig.inputs {
        let syn::FnArg::Typed(argument) = argument else {
            continue;
        };
        let syn::Pat::Ident(name) = argument.pat.as_ref() else {
            continue;
        };
        match parameter_flavor_in(stage, &argument.ty) {
            ParameterFlavor::Invocation => {
                invocation.get_or_insert_with(|| name.ident.to_string());
                if name.mutability.is_some() {
                    found.exclude(Why::MutableInvocation, argument, "makes mutable");
                }
            }
            ParameterFlavor::LocalId | ParameterFlavor::GroupId => {
                found.exclude(Why::InvocationBuiltin, argument, "");
            }
            ParameterFlavor::Workgroup => found.exclude(Why::WorkgroupMemory, argument, ""),
            ParameterFlavor::StorageReadWrite => outputs.push(Output {
                name: name.ident.to_string(),
                argument,
            }),
            ParameterFlavor::Value if is_storage_texture(&argument.ty) => {
                found.exclude(Why::StorageTexture, argument, "");
            }
            _ => {}
        }
    }

    // The body.
    let mut scan = BodyScan {
        id: invocation.clone().unwrap_or_default(),
        ..BodyScan::default()
    };
    scan.visit_block(&function.block);
    for barrier in &scan.barriers {
        found.exclude(Why::Barrier, barrier, "");
    }
    for atomic in &scan.atomics {
        found.exclude(Why::Atomic, atomic, "");
    }
    for statement in &scan.returns {
        found.exclude(Why::Return, statement, "");
    }
    for assignment in &scan.id_assignments {
        found.exclude(Why::MutableInvocation, assignment, "assigns to");
    }

    // The own index.
    let mut alias = None;
    if let Some((first, rest)) = scan.aliases.split_first() {
        for second in rest {
            found.exclude(
                Why::OwnIndexAlias,
                &second.statement,
                "is a second alias of the own index",
            );
        }
        let bound_again = scan
            .bindings
            .iter()
            .filter(|(name, _)| *name == first.name)
            .nth(1);
        if first.mutable {
            found.exclude(
                Why::OwnIndexAlias,
                &first.statement,
                "is a `mut` alias of the own index",
            );
        } else if let Some((_, binding)) = bound_again {
            found.exclude(
                Why::OwnIndexAlias,
                binding,
                &format!(
                    "binds `{}` again, and it is the alias of the own index",
                    first.name
                ),
            );
        } else {
            alias = Some(first.name.clone());
        }
    }
    let own = Own {
        id: scan.id.clone(),
        alias,
    };

    // The outputs. A buffer the atomics use is excluded as an atomic buffer already.
    let atomic_buffers = atomic_buffer_names(function);
    let mut written: Vec<&Output<'_>> = Vec::new();
    let mut own_written: Vec<&Output<'_>> = Vec::new();
    for output in &outputs {
        if atomic_buffers.contains(&output.name) {
            continue;
        }
        let mut uses = OutputScan::new(&output.name, &own);
        uses.visit_block(&function.block);
        for finding in &uses.findings {
            found.exclude(finding.why, &finding.tokens, &format!("`{}`", output.name));
        }
        if uses.written {
            written.push(output);
            if uses.own_writes > 0 {
                own_written.push(output);
            }
        } else if uses.findings.is_empty() {
            found.exclude(
                Why::OutputUnwritten,
                output.argument,
                &format!("`{}`", output.name),
            );
        }
    }

    // Every output is definitely written.
    let names: HashSet<String> = written.iter().map(|output| output.name.clone()).collect();
    for output in own_written {
        let definite = Definite {
            output: &output.name,
            own: &own,
            outputs: &names,
        };
        if !definite.block(&function.block) {
            let blame = definite
                .blame_block(&function.block)
                .unwrap_or_else(|| output.argument.to_token_stream());
            found.exclude(Why::ConditionalWrite, &blame, &format!("`{}`", output.name));
        }
    }

    // At least one output, each with a colour-target format, within D62's bound.
    if written.is_empty() {
        found.exclude(Why::NoOutput, &function.sig.ident, "");
    }
    let mut bytes = 0_u32;
    for (index, output) in written.iter().enumerate() {
        match wrapper_element_type(&output.argument.ty)
            .ok()
            .and_then(target_bytes)
        {
            Some(size) => bytes += size,
            None => {
                let element = wrapper_element_type(&output.argument.ty).map_or_else(
                    |_| "an element of no known type".to_owned(),
                    |ty| format!("the element type `{}`", render(&ty.to_token_stream())),
                );
                found.exclude(Why::TargetElement, output.argument, &element);
            }
        }
        if index == MAX_TARGETS {
            found.exclude(Why::TargetCount, output.argument, &(index + 1).to_string());
        }
    }
    if bytes > MAX_TARGET_BYTES
        && let Some(last) = written.last()
    {
        found.exclude(Why::TargetBytes, last.argument, &bytes.to_string());
    }

    let mut shapes = vec![Shape::Grid];
    if found.exclusions.is_empty() {
        shapes.push(Shape::Pixel);
    }
    Classification {
        shapes,
        exclusions: found.exclusions,
    }
}

/// Refuses a pin that the classification does not allow: the error is spanned over the pin, carries the code of the
/// reason, and names the construct that rules the shape out (plan 5.3).
pub(crate) fn check_pin(
    function: &syn::ItemFn,
    classification: &Classification,
    pin: &ShapePin,
) -> syn::Result<()> {
    if classification.allows(pin.shape) {
        return Ok(());
    }
    let Some(exclusion) = classification.first_exclusion(pin.shape) else {
        return Ok(());
    };
    Err(diagnostics::error(
        exclusion.why.code(),
        &pin.tokens,
        exclusion.message(&function.sig.ident.to_string()),
    ))
}

// ---------------------------------------------------------------------------------------------------------------
// Collecting.
// ---------------------------------------------------------------------------------------------------------------

#[derive(Default)]
struct Found {
    exclusions: Vec<Exclusion>,
}

impl Found {
    fn exclude(&mut self, why: Why, node: impl ToTokens, detail: &str) {
        let tokens = node.to_token_stream();
        self.exclusions.push(Exclusion {
            shape: Shape::Pixel,
            why,
            construct: render(&tokens),
            detail: detail.to_owned(),
        });
    }
}

/// A `StorageMut` parameter.
struct Output<'a> {
    name: String,
    argument: &'a syn::PatType,
}

/// The source text of `tokens`, tidied (`out [i + 1]` to `out[i + 1]`) and cut to a line.
fn render(tokens: &TokenStream) -> String {
    let mut text = tokens.to_string();
    for (from, to) in [
        (" . ", "."),
        (" .", "."),
        (". ", "."),
        (" [", "["),
        ("[ ", "["),
        (" ]", "]"),
        (" (", "("),
        ("( ", "("),
        (" )", ")"),
        (" ,", ","),
        (" ;", ";"),
        (" :: ", "::"),
        (":: ", "::"),
        (" ::", "::"),
        ("& mut ", "&mut "),
        ("& ", "&"),
        ("! ", "!"),
    ] {
        text = text.replace(from, to);
    }
    const LIMIT: usize = 72;
    if text.chars().count() > LIMIT {
        let cut: String = text.chars().take(LIMIT).collect();
        text = format!("{}...", cut.trim_end());
    }
    text
}

/// Whether a value parameter has a storage-texture type (`StorageTexture2D<F>`, `WriteTexture2D<F>`).
fn is_storage_texture(ty: &syn::Type) -> bool {
    matches!(
        crate::slang::type_name(ty).as_deref(),
        Some("StorageTexture2D" | "WriteTexture2D")
    )
}

/// The bytes of the colour target an output element becomes, or `None` when it has no target format (D62).
fn target_bytes(element: &syn::Type) -> Option<u32> {
    let syn::Type::Path(path) = element else {
        return None;
    };
    match path.path.get_ident()?.unraw().to_string().as_str() {
        "u32" | "uint" | "i32" | "int" | "f32" | "float" => Some(4),
        "Vec2" | "IVec2" | "UVec2" => Some(8),
        "Vec4" | "IVec4" | "UVec4" => Some(16),
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Expression helpers.
// ---------------------------------------------------------------------------------------------------------------

/// `expression` without the parentheses and groups around it.
fn peel(mut expression: &syn::Expr) -> &syn::Expr {
    loop {
        match expression {
            syn::Expr::Paren(paren) => expression = &paren.expr,
            syn::Expr::Group(group) => expression = &group.expr,
            other => return other,
        }
    }
}

/// Whether `expression` is the plain name `name`.
fn is_name(expression: &syn::Expr, name: &str) -> bool {
    matches!(peel(expression), syn::Expr::Path(path) if path.qself.is_none() && path.path.is_ident(name))
}

/// Whether `expression` is `id.x` or `id.global_id().x` for the invocation parameter `id`.
fn is_direct_own(expression: &syn::Expr, id: &str) -> bool {
    let syn::Expr::Field(field) = peel(expression) else {
        return false;
    };
    if !matches!(&field.member, syn::Member::Named(member) if member == "x") {
        return false;
    }
    match peel(&field.base) {
        syn::Expr::Path(path) => path.qself.is_none() && path.path.is_ident(id),
        syn::Expr::MethodCall(call) => {
            call.method == "global_id"
                && call.args.is_empty()
                && call.turbofish.is_none()
                && is_name(&call.receiver, id)
        }
        _ => false,
    }
}

/// The name an expression is rooted in through field accesses and indexing: `id` for `id.x`.
fn root_name(expression: &syn::Expr) -> Option<String> {
    match peel(expression) {
        syn::Expr::Path(path) if path.qself.is_none() => {
            path.path.get_ident().map(ToString::to_string)
        }
        syn::Expr::Field(field) => root_name(&field.base),
        syn::Expr::Index(index) => root_name(&index.expr),
        _ => None,
    }
}

/// Whether `op` is a compound assignment (`+=` and the rest).
fn is_compound(op: &syn::BinOp) -> bool {
    matches!(
        op,
        syn::BinOp::AddAssign(_)
            | syn::BinOp::SubAssign(_)
            | syn::BinOp::MulAssign(_)
            | syn::BinOp::DivAssign(_)
            | syn::BinOp::RemAssign(_)
            | syn::BinOp::BitXorAssign(_)
            | syn::BinOp::BitAndAssign(_)
            | syn::BinOp::BitOrAssign(_)
            | syn::BinOp::ShlAssign(_)
            | syn::BinOp::ShrAssign(_)
    )
}

/// The element index of the own-index spellings the kernel uses.
struct Own {
    /// The name of the invocation parameter.
    id: String,
    /// The one immutable `let i = id.x;` the classifier accepted, if any.
    alias: Option<String>,
}

impl Own {
    fn is(&self, expression: &syn::Expr) -> bool {
        is_direct_own(expression, &self.id)
            || self
                .alias
                .as_deref()
                .is_some_and(|alias| is_name(expression, alias))
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The scan of the whole body.
// ---------------------------------------------------------------------------------------------------------------

/// A `let name = id.x;`.
struct Alias {
    name: String,
    mutable: bool,
    statement: TokenStream,
}

/// What the body holds that rules `pixel` out whatever the outputs do, and the facts the own index needs.
#[derive(Default)]
struct BodyScan {
    id: String,
    returns: Vec<TokenStream>,
    barriers: Vec<TokenStream>,
    atomics: Vec<TokenStream>,
    aliases: Vec<Alias>,
    /// Every binding of a name in the body, in source order, with the pattern.
    bindings: Vec<(String, TokenStream)>,
    id_assignments: Vec<TokenStream>,
}

impl<'ast> Visit<'ast> for BodyScan {
    fn visit_expr_return(&mut self, expression: &'ast syn::ExprReturn) {
        self.returns.push(expression.to_token_stream());
        visit::visit_expr_return(self, expression);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if is_barrier_call(call) {
            self.barriers.push(call.to_token_stream());
        } else if let syn::Expr::Path(path) = call.func.as_ref()
            && atomic_operation_name(path).is_some()
        {
            self.atomics.push(call.to_token_stream());
        }
        visit::visit_expr_call(self, call);
    }

    fn visit_expr_assign(&mut self, assignment: &'ast syn::ExprAssign) {
        if root_name(&assignment.left).is_some_and(|name| name == self.id) {
            self.id_assignments.push(assignment.to_token_stream());
        }
        visit::visit_expr_assign(self, assignment);
    }

    fn visit_expr_binary(&mut self, binary: &'ast syn::ExprBinary) {
        if is_compound(&binary.op) && root_name(&binary.left).is_some_and(|name| name == self.id) {
            self.id_assignments.push(binary.to_token_stream());
        }
        visit::visit_expr_binary(self, binary);
    }

    fn visit_local(&mut self, local: &'ast syn::Local) {
        if let Some(initial) = &local.init
            && is_direct_own(&initial.expr, &self.id)
            && let Ok((pattern, _)) = local_binding(&local.pat)
        {
            self.aliases.push(Alias {
                name: pattern.ident.to_string(),
                mutable: pattern.mutability.is_some(),
                statement: local.to_token_stream(),
            });
        }
        visit::visit_local(self, local);
    }

    fn visit_pat_ident(&mut self, pattern: &'ast syn::PatIdent) {
        self.bindings
            .push((pattern.ident.to_string(), pattern.to_token_stream()));
        visit::visit_pat_ident(self, pattern);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The scan of one output.
// ---------------------------------------------------------------------------------------------------------------

struct Finding {
    why: Why,
    tokens: TokenStream,
}

/// Every use of one output buffer: which are `out[own] = v` and `out.len()`, and which are something else.
struct OutputScan<'a> {
    name: &'a str,
    own: &'a Own,
    /// The output is the target of some assignment.
    written: bool,
    /// How many assignments are `out[own] = v`.
    own_writes: usize,
    findings: Vec<Finding>,
}

/// What an assignment target is, as far as one output goes.
enum Target<'ast> {
    /// `out[index]`.
    Element(&'ast syn::Expr),
    /// `out[index].component`.
    Component(&'ast syn::Expr),
    /// Anything else.
    Other,
}

impl<'a> OutputScan<'a> {
    fn new(name: &'a str, own: &'a Own) -> Self {
        Self {
            name,
            own,
            written: false,
            own_writes: 0,
            findings: Vec::new(),
        }
    }

    fn find(&mut self, why: Why, node: impl ToTokens) {
        self.findings.push(Finding {
            why,
            tokens: node.to_token_stream(),
        });
    }

    fn target<'ast>(&self, expression: &'ast syn::Expr) -> Target<'ast> {
        match peel(expression) {
            syn::Expr::Index(index) if is_name(&index.expr, self.name) => {
                Target::Element(&index.index)
            }
            syn::Expr::Field(field) => {
                let mut base = peel(&field.base);
                while let syn::Expr::Field(inner) = base {
                    base = peel(&inner.base);
                }
                match base {
                    syn::Expr::Index(index) if is_name(&index.expr, self.name) => {
                        Target::Component(&index.index)
                    }
                    _ => Target::Other,
                }
            }
            _ => Target::Other,
        }
    }
}

impl<'ast> Visit<'ast> for OutputScan<'_> {
    fn visit_expr_assign(&mut self, assignment: &'ast syn::ExprAssign) {
        match self.target(&assignment.left) {
            Target::Element(index) => {
                self.written = true;
                if self.own.is(index) {
                    self.own_writes += 1;
                } else {
                    self.find(Why::OwnIndexWrite, &assignment.left);
                }
                self.visit_expr(index);
                self.visit_expr(&assignment.right);
            }
            Target::Component(index) => {
                self.written = true;
                self.find(Why::OutputComponent, &assignment.left);
                self.visit_expr(index);
                self.visit_expr(&assignment.right);
            }
            Target::Other => visit::visit_expr_assign(self, assignment),
        }
    }

    fn visit_expr_binary(&mut self, binary: &'ast syn::ExprBinary) {
        if is_compound(&binary.op) {
            let (why, index) = match self.target(&binary.left) {
                Target::Element(index) => (Why::OutputCompound, Some(index)),
                Target::Component(index) => (Why::OutputComponent, Some(index)),
                Target::Other => (Why::OutputCompound, None),
            };
            if let Some(index) = index {
                self.written = true;
                self.find(why, binary);
                self.visit_expr(index);
                self.visit_expr(&binary.right);
                return;
            }
        }
        visit::visit_expr_binary(self, binary);
    }

    fn visit_expr_index(&mut self, index: &'ast syn::ExprIndex) {
        if is_name(&index.expr, self.name) {
            self.find(Why::OutputRead, index);
            self.visit_expr(&index.index);
            return;
        }
        visit::visit_expr_index(self, index);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if call.method == "len" && call.args.is_empty() && is_name(&call.receiver, self.name) {
            return;
        }
        visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        if path.qself.is_none() && path.path.is_ident(self.name) {
            self.find(Why::OutputOther, path);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Definite assignment.
// ---------------------------------------------------------------------------------------------------------------

/// Whether the body definitely writes `output` at the own index, by ordinary definite assignment with the one
/// transparency of plan R1.1 M1.
struct Definite<'a> {
    output: &'a str,
    own: &'a Own,
    /// The names of the outputs, for the length in a transparent guard.
    outputs: &'a HashSet<String>,
}

impl Definite<'_> {
    fn block(&self, block: &syn::Block) -> bool {
        block
            .stmts
            .iter()
            .any(|statement| self.statement(statement))
    }

    fn statement(&self, statement: &syn::Stmt) -> bool {
        match statement {
            syn::Stmt::Expr(expression, _) => self.expression(expression),
            syn::Stmt::Local(local) => local
                .init
                .as_ref()
                .is_some_and(|initial| self.expression(&initial.expr)),
            _ => false,
        }
    }

    fn expression(&self, expression: &syn::Expr) -> bool {
        match peel(expression) {
            syn::Expr::Block(block) => self.block(&block.block),
            syn::Expr::Assign(assignment) => {
                self.is_own_write(&assignment.left) || self.expression(&assignment.right)
            }
            syn::Expr::If(branch) => match &branch.else_branch {
                Some((_, other)) => self.block(&branch.then_branch) && self.expression(other),
                None => self.transparent(&branch.cond) && self.block(&branch.then_branch),
            },
            syn::Expr::Match(arms) => {
                !arms.arms.is_empty() && arms.arms.iter().all(|arm| self.expression(&arm.body))
            }
            _ => false,
        }
    }

    /// `output[own]`, as an assignment target.
    fn is_own_write(&self, target: &syn::Expr) -> bool {
        matches!(peel(target), syn::Expr::Index(index)
            if is_name(&index.expr, self.output) && self.own.is(&index.index))
    }

    /// Whether `condition` is always true for a live pixel: every conjunct compares the own index with the length of
    /// an output.
    fn transparent(&self, condition: &syn::Expr) -> bool {
        match peel(condition) {
            syn::Expr::Binary(binary) => match binary.op {
                syn::BinOp::And(_) => {
                    self.transparent(&binary.left) && self.transparent(&binary.right)
                }
                syn::BinOp::Lt(_) => {
                    self.own.is(&binary.left) && self.is_output_length(&binary.right)
                }
                syn::BinOp::Gt(_) => {
                    self.is_output_length(&binary.left) && self.own.is(&binary.right)
                }
                _ => false,
            },
            _ => false,
        }
    }

    fn is_output_length(&self, expression: &syn::Expr) -> bool {
        matches!(peel(expression), syn::Expr::MethodCall(call)
            if call.method == "len"
                && call.args.is_empty()
                && matches!(peel(&call.receiver), syn::Expr::Path(path)
                    if path.qself.is_none()
                        && path.path.get_ident().is_some_and(|name| self.outputs.contains(&name.to_string()))))
    }

    // The construct to name when the body does not definitely write the output.

    fn blame_block(&self, block: &syn::Block) -> Option<TokenStream> {
        block.stmts.iter().find_map(|statement| match statement {
            syn::Stmt::Expr(expression, _) => self.blame(expression),
            syn::Stmt::Local(local) => local
                .init
                .as_ref()
                .and_then(|initial| self.blame(&initial.expr)),
            _ => None,
        })
    }

    fn blame(&self, expression: &syn::Expr) -> Option<TokenStream> {
        match peel(expression) {
            syn::Expr::Block(block) => self.blame_block(&block.block),
            syn::Expr::If(branch) => {
                let writes = self.writes_in_block(&branch.then_branch)
                    || branch
                        .else_branch
                        .as_ref()
                        .is_some_and(|(_, other)| self.writes_in(other));
                if !writes {
                    return None;
                }
                match &branch.else_branch {
                    None if self.transparent(&branch.cond) => self.blame_block(&branch.then_branch),
                    None => self.opaque_conjunct(&branch.cond),
                    Some((_, other)) => {
                        if self.block(&branch.then_branch) && self.expression(other) {
                            None
                        } else {
                            self.blame_block(&branch.then_branch)
                                .or_else(|| self.blame(other))
                                .or_else(|| Some(branch.cond.to_token_stream()))
                        }
                    }
                }
            }
            syn::Expr::Match(arms) => {
                if !arms.arms.iter().any(|arm| self.writes_in(&arm.body)) {
                    return None;
                }
                arms.arms
                    .iter()
                    .find_map(|arm| self.blame(&arm.body))
                    .or_else(|| Some(arms.expr.to_token_stream()))
            }
            syn::Expr::ForLoop(looped) if self.writes_in_block(&looped.body) => {
                let (pattern, iterable) = (&looped.pat, &looped.expr);
                Some(quote!(for #pattern in #iterable))
            }
            _ => None,
        }
    }

    /// The first conjunct of `condition` that is not a transparent guard.
    fn opaque_conjunct(&self, condition: &syn::Expr) -> Option<TokenStream> {
        match peel(condition) {
            syn::Expr::Binary(binary) if matches!(binary.op, syn::BinOp::And(_)) => self
                .opaque_conjunct(&binary.left)
                .or_else(|| self.opaque_conjunct(&binary.right)),
            other if self.transparent(other) => None,
            other => Some(other.to_token_stream()),
        }
    }

    fn writes_in(&self, expression: &syn::Expr) -> bool {
        let mut finder = WriteFinder {
            definite: self,
            found: false,
        };
        finder.visit_expr(expression);
        finder.found
    }

    fn writes_in_block(&self, block: &syn::Block) -> bool {
        let mut finder = WriteFinder {
            definite: self,
            found: false,
        };
        finder.visit_block(block);
        finder.found
    }
}

/// Whether some assignment to `output[own]` stands anywhere inside.
struct WriteFinder<'a, 'b> {
    definite: &'a Definite<'b>,
    found: bool,
}

impl<'ast> Visit<'ast> for WriteFinder<'_, '_> {
    fn visit_expr_assign(&mut self, assignment: &'ast syn::ExprAssign) {
        if self.definite.is_own_write(&assignment.left) {
            self.found = true;
        }
        visit::visit_expr_assign(self, assignment);
    }
}

#[cfg(test)]
mod tests;
