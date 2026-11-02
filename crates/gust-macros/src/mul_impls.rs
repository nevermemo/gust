//! The `Mul` and `MulAssign` impls of `gust::math`, generated from the product table (slice m6.m2).
//!
//! `crates/gust/src/math.rs` invokes [`gust_mul_impls!`](crate::gust_mul_impls) once and writes no product by hand
//! (a source test pins that), so the Rust operators and the Slang overloads `__gust_mul` of
//! [`crate::slang::mul_table`] are one list. Every function is `#[inline]`, as the rest of `gust::math` (its module
//! documentation explains why: the overflow rule of the integer vectors depends on it).

use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};

use crate::slang::mul_table::{self, Kind, MUL_PAIRS, Row};

/// The impls of every row of the product table except the scalar ones.
pub(crate) fn mul_impls() -> TokenStream {
    let mut tokens = TokenStream::new();
    for row in MUL_PAIRS.iter().filter(|row| row.kind != Kind::Scalar) {
        tokens.extend(mul_impl(row));
        if row.assigns() {
            tokens.extend(mul_assign_impl(row));
        }
    }
    tokens
}

fn ident(name: &str) -> Ident {
    format_ident!("{}", name)
}

/// The component names of a vector spelling.
fn fields(name: &str) -> Vec<Ident> {
    ["x", "y", "z", "w"][..mul_table::width(name)]
        .iter()
        .map(|field| ident(field))
        .collect()
}

/// `impl Mul<right> for left` for a row.
fn mul_impl(row: &Row) -> TokenStream {
    let (left, right, output) = (ident(row.left), ident(row.right), ident(row.output));
    let width = mul_table::width(row.left);
    let body = match (
        row.kind,
        mul_table::is_matrix(row.left),
        mul_table::is_matrix(row.right),
        mul_table::is_scalar(row.left),
        mul_table::is_scalar(row.right),
    ) {
        // The matrix products are the methods of the same meaning: `m.mul_vecN(v)` is M x v and `a.mul_matN(b)` is
        // a x b, summed in written order.
        (Kind::MatMul, _, false, _, _) => {
            let method = format_ident!("mul_vec{}", width);
            quote!(self.#method(rhs))
        }
        (Kind::MatMul, _, true, _, _) => {
            let method = format_ident!("mul_mat{}", width);
            quote!(self.#method(rhs))
        }
        (_, true, _, _, true) => quote!(Self {
            cols: self.cols.map(|column| column * rhs)
        }),
        (_, _, true, true, _) => quote!(#output { cols: rhs.cols.map(|column| self * column) }),
        (_, _, _, false, true) => {
            let fields = fields(row.left);
            quote!(#output { #(#fields: self.#fields * rhs),* })
        }
        (_, _, _, true, false) => {
            let fields = fields(row.right);
            quote!(#output { #(#fields: self * rhs.#fields),* })
        }
        _ => {
            let fields = fields(row.left);
            quote!(#output { #(#fields: self.#fields * rhs.#fields),* })
        }
    };
    quote! {
        impl ::core::ops::Mul<#right> for #left {
            type Output = #output;
            #[inline]
            fn mul(self, rhs: #right) -> #output {
                #body
            }
        }
    }
}

/// `impl MulAssign<right> for left` for a row whose output is its left type.
fn mul_assign_impl(row: &Row) -> TokenStream {
    let (left, right) = (ident(row.left), ident(row.right));
    let body = if mul_table::is_matrix(row.left) {
        quote! {
            for column in &mut self.cols {
                *column *= rhs;
            }
        }
    } else {
        let fields = fields(row.left);
        if mul_table::is_scalar(row.right) {
            quote!(#(self.#fields *= rhs;)*)
        } else {
            quote!(#(self.#fields *= rhs.#fields;)*)
        }
    };
    quote! {
        impl ::core::ops::MulAssign<#right> for #left {
            #[inline]
            fn mul_assign(&mut self, rhs: #right) {
                #body
            }
        }
    }
}
