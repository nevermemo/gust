//! The conservative call graph of a GPU module (M9 plan 1 item 3, D162; slice m9.a2).
//!
//! **Nodes** are the functions that have a body: every helper, every kernel, and every function of an `impl` block,
//! inherent or a trait impl (a method or an associated function), one node per function and impl. A `trait` declaration
//! has no body and no node (slice m9.g1): a call of a trait method is a call of the methods of that name in the impls.
//! **Edges** are drawn from every call in a body to every node the call could resolve to, with no type inference beyond
//! what the dialect already knows:
//!
//! - `f(..)` is an edge to the helper `f` (a kernel is not callable, so no edge goes to one);
//! - `x.m(..)` is an edge to **every** method named `m` of every `impl` block, trait impls included, whatever the type
//!   of `x` is (a trait path call `T::m(&x)` is the same call, written `x.m(..)` before the graph runs);
//! - `S::f(..)` and `Self::f(..)` are edges to every function `f` of the `impl` blocks of `S` (`Self` is the struct of the
//!   enclosing `impl`).
//!
//! Over-approximation is the point: an edge that Rust would not take can make a cycle, and a missing edge would let
//! recursion through. A cycle is GUST-E0120, reported at the call that closes it. The graph runs after the validation of
//! every item, so the bodies it reads are already restricted to the dialect.
//!
//! The graph does not carry uniformity. That stays the rule of D154: every callee is validated as a helper (value
//! parameters only, no barrier, subgroup operation, atomic or storage access), so no path through the graph reaches a
//! construct that the local rules of the kernel keep out.

use std::collections::HashMap;

use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::ext::IdentExt as _;
use syn::visit::{self, Visit};

use super::{impl_self_name, receiver_of};
use crate::diagnostics::{self, Code};

/// One call: the node it may reach, and the tokens the diagnostic points at.
struct Edge {
    to: usize,
    at: TokenStream,
}

/// A function with a body.
struct Node<'a> {
    /// How the diagnostic names it: `f` for a helper or kernel, `S::m` for a function of an `impl`.
    label: String,
    /// The struct of the enclosing `impl`, the meaning of `Self` in the body.
    owner: Option<String>,
    block: &'a syn::Block,
}

/// What a call can resolve to: the lookup tables of the graph.
#[derive(Default)]
struct Names {
    /// Helpers (the functions that are not kernels) by name.
    helpers: HashMap<String, usize>,
    /// Methods (functions with a receiver) by name, across every `impl`.
    methods: HashMap<String, Vec<usize>>,
    /// Every function of an `impl` by `Struct::name`.
    members: HashMap<String, Vec<usize>>,
}

/// Refuses a module whose call graph has a cycle (GUST-E0120).
pub(super) fn check(items: &[syn::Item]) -> syn::Result<()> {
    let mut nodes: Vec<Node<'_>> = Vec::new();
    let mut names = Names::default();
    for item in items {
        match item {
            syn::Item::Fn(function) => {
                let kernel = function
                    .attrs
                    .iter()
                    .any(|attribute| attribute.path().is_ident("kernel"));
                let name = function.sig.ident.unraw().to_string();
                if !kernel {
                    names.helpers.insert(name.clone(), nodes.len());
                }
                nodes.push(Node {
                    label: name,
                    owner: None,
                    block: &function.block,
                });
            }
            syn::Item::Impl(block) => {
                let Some(owner) = impl_self_name(block) else {
                    continue;
                };
                for inner in &block.items {
                    let syn::ImplItem::Fn(method) = inner else {
                        continue;
                    };
                    let name = method.sig.ident.unraw().to_string();
                    let index = nodes.len();
                    if receiver_of(&method.sig).is_some() {
                        names.methods.entry(name.clone()).or_default().push(index);
                    }
                    names
                        .members
                        .entry(format!("{owner}::{name}"))
                        .or_default()
                        .push(index);
                    nodes.push(Node {
                        label: format!("{owner}::{name}"),
                        owner: Some(owner.clone()),
                        block: &method.block,
                    });
                }
            }
            _ => {}
        }
    }

    let edges: Vec<Vec<Edge>> = nodes
        .iter()
        .map(|node| {
            let mut calls = Calls {
                owner: node.owner.as_deref(),
                names: &names,
                edges: Vec::new(),
            };
            calls.visit_block(node.block);
            calls.edges
        })
        .collect();

    let mut state = vec![State::Unvisited; nodes.len()];
    let mut path = Vec::new();
    for start in 0..nodes.len() {
        if state[start] == State::Unvisited
            && let Some((cycle, at)) = find_cycle(start, &edges, &mut state, &mut path)
        {
            let chain: Vec<&str> = cycle
                .iter()
                .map(|&index| nodes[index].label.as_str())
                .collect();
            let by_name = if cycle.iter().any(|&index| nodes[index].owner.is_some()) {
                " The graph is built from names without types: a call `x.m(..)` counts as a call of every method named `m`, \
                 so a method that calls another struct's method of the same name is a cycle too; rename one of them."
            } else {
                ""
            };
            let message = if let [only, _] = cycle.as_slice() {
                format!(
                    "`{}` calls itself ({} -> {}): recursion has no GPU lowering, and a loop does the same work; rewrite it as a `for` loop.{by_name}",
                    nodes[*only].label, chain[0], chain[1]
                )
            } else {
                format!(
                    "these functions call each other in a cycle ({}): recursion has no GPU lowering; rewrite it as a `for` loop.{by_name}",
                    chain.join(" -> ")
                )
            };
            return Err(diagnostics::error(Code::RecursiveCall, at, message));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Unvisited,
    /// On the current path.
    Open,
    Done,
}

/// Depth-first search from `node`: the first back edge found, as the cycle (the nodes of the path from the target to the
/// source, then the target again) and the call that closes it. Edges are visited in source order, so the report is
/// deterministic.
fn find_cycle(
    node: usize,
    edges: &[Vec<Edge>],
    state: &mut [State],
    path: &mut Vec<usize>,
) -> Option<(Vec<usize>, TokenStream)> {
    state[node] = State::Open;
    path.push(node);
    for edge in &edges[node] {
        match state[edge.to] {
            State::Open => {
                let start = path.iter().position(|&index| index == edge.to)?;
                let mut cycle = path[start..].to_vec();
                cycle.push(edge.to);
                return Some((cycle, edge.at.clone()));
            }
            State::Unvisited => {
                if let Some(found) = find_cycle(edge.to, edges, state, path) {
                    return Some(found);
                }
            }
            State::Done => {}
        }
    }
    path.pop();
    state[node] = State::Done;
    None
}

/// The edges out of one body.
struct Calls<'a> {
    owner: Option<&'a str>,
    names: &'a Names,
    edges: Vec<Edge>,
}

impl Calls<'_> {
    fn add(&mut self, targets: &[usize], at: &impl ToTokens) {
        for &to in targets {
            self.edges.push(Edge {
                to,
                at: at.to_token_stream(),
            });
        }
    }
}

impl<'ast> Visit<'ast> for Calls<'_> {
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = call.func.as_ref()
            && path.qself.is_none()
            && path.path.leading_colon.is_none()
        {
            if let Some(ident) = path.path.get_ident() {
                if let Some(&to) = self.names.helpers.get(&ident.unraw().to_string()) {
                    self.add(&[to], ident);
                }
            } else if path.path.segments.len() == 2 {
                let first = path.path.segments[0].ident.unraw().to_string();
                let member = path.path.segments[1].ident.unraw().to_string();
                let owner = if first == "Self" {
                    self.owner.map(str::to_owned)
                } else {
                    Some(first)
                };
                if let Some(owner) = owner
                    && let Some(targets) = self.names.members.get(&format!("{owner}::{member}"))
                {
                    let targets = targets.clone();
                    self.add(&targets, path);
                }
            }
        }
        visit::visit_expr_call(self, call);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if let Some(targets) = self.names.methods.get(&call.method.unraw().to_string()) {
            let targets = targets.clone();
            self.add(&targets, &call.method);
        }
        visit::visit_expr_method_call(self, call);
    }
}
