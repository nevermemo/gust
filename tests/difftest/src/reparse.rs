//! Test support: a re-parser for the emitted kernel text, written with Rust's real operator precedence.
//!
//! `reparse(emit_module(p))` must give back exactly the body and the result of `p`. This is the check that catches an
//! emitter that drops a parenthesis: rustc would accept `a << b + c` (it reads `a << (b + c)`), so neither the
//! `#[gpu]` validator nor `cargo check` can see the difference; the AST comparison can.
//!
//! The parser reads only the grammar the emitter writes: `let`, assignment, `if`/`else`, `for v in 0u32..Nu32`, the
//! final `out[i] = ...` store, and expressions with parentheses, unary `-`/`!`, `as`, the binary operators and
//! `name[...]` / `name.len()`. A negated integer literal (`-5i32`) is read as one literal, as `Expr::negated` folds it.

use std::collections::BTreeMap;

use crate::ast::{BinOp, Expr, InputBuf, Stmt, Ty, UnOp};
use crate::emit::BUFFER_NAMES;

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Ident(String),
    Int(u64, String),
    Punct(&'static str),
}

const PUNCT: [&str; 33] = [
    "::", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||", "..", "(", ")", "[", "]", "{", "}", ";",
    ":", ",", ".", "=", "<", ">", "+", "-", "*", "/", "%", "&", "|", "^", "!", "#",
];

/// The `out[i] = ...` store, as an assignment to this variable id.
const OUT: u32 = u32::MAX;

fn tokenize(text: &str) -> Result<Vec<Tok>, String> {
    let chars: Vec<char> = text.chars().collect();
    let mut toks = Vec::new();
    let mut pos = 0;
    while pos < chars.len() {
        let c = chars[pos];
        if c.is_whitespace() {
            pos += 1;
        } else if c == '/' && chars.get(pos + 1) == Some(&'/') {
            while pos < chars.len() && chars[pos] != '\n' {
                pos += 1;
            }
        } else if c.is_ascii_digit() {
            let start = pos;
            while pos < chars.len() && chars[pos].is_ascii_digit() {
                pos += 1;
            }
            let digits: String = chars[start..pos].iter().collect();
            let value = digits
                .parse::<u64>()
                .map_err(|e| format!("{digits}: {e}"))?;
            let suffix_start = pos;
            while pos < chars.len() && (chars[pos].is_ascii_alphanumeric() || chars[pos] == '_') {
                pos += 1;
            }
            toks.push(Tok::Int(value, chars[suffix_start..pos].iter().collect()));
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = pos;
            while pos < chars.len() && (chars[pos].is_ascii_alphanumeric() || chars[pos] == '_') {
                pos += 1;
            }
            toks.push(Tok::Ident(chars[start..pos].iter().collect()));
        } else {
            let rest: String = chars[pos..chars.len().min(pos + 2)].iter().collect();
            let punct = PUNCT
                .iter()
                .find(|p| rest.starts_with(**p))
                .ok_or_else(|| format!("unexpected character {c:?}"))?;
            toks.push(Tok::Punct(punct));
            pos += punct.len();
        }
    }
    Ok(toks)
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
    vars: BTreeMap<u32, Ty>,
    inputs: Vec<Ty>,
}

type Res<T> = Result<T, String>;

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn next(&mut self) -> Res<Tok> {
        let tok = self.toks.get(self.pos).cloned();
        self.pos += 1;
        tok.ok_or_else(|| "unexpected end of text".to_owned())
    }

    fn eat_punct(&mut self, punct: &'static str) -> bool {
        if self.peek() == Some(&Tok::Punct(punct)) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect_punct(&mut self, punct: &'static str) -> Res<()> {
        if self.eat_punct(punct) {
            Ok(())
        } else {
            Err(format!(
                "expected `{punct}` at token {}: {:?}",
                self.pos,
                self.peek()
            ))
        }
    }

    fn eat_ident(&mut self, ident: &str) -> bool {
        if matches!(self.peek(), Some(Tok::Ident(name)) if name == ident) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn ident(&mut self) -> Res<String> {
        match self.next()? {
            Tok::Ident(name) => Ok(name),
            other => Err(format!("expected an identifier, found {other:?}")),
        }
    }

    fn var_id(name: &str) -> Option<u32> {
        name.strip_prefix('v')?.parse().ok()
    }

    fn ty(&mut self) -> Res<Ty> {
        match self.ident()?.as_str() {
            "u32" => Ok(Ty::U32),
            "i32" => Ok(Ty::I32),
            "bool" => Ok(Ty::Bool),
            other => Err(format!("unknown type {other}")),
        }
    }

    fn binop(&self) -> Option<BinOp> {
        let Some(Tok::Punct(punct)) = self.peek() else {
            return None;
        };
        BinOp::ALL.iter().copied().find(|op| op.symbol() == *punct)
    }

    fn expr(&mut self, min_prec: u8) -> Res<Expr> {
        let mut lhs = self.cast()?;
        while let Some(op) = self.binop() {
            let prec = op.precedence();
            if prec < min_prec {
                break;
            }
            self.pos += 1;
            let rhs = self.expr(prec + 1)?;
            lhs = Expr::bin(op, lhs, rhs);
            if op.is_comparison() && self.binop().is_some_and(BinOp::is_comparison) {
                return Err("chained comparison".into());
            }
        }
        Ok(lhs)
    }

    fn cast(&mut self) -> Res<Expr> {
        let mut e = self.unary()?;
        while self.eat_ident("as") {
            let to = self.ty()?;
            e = Expr::cast(to, e);
        }
        Ok(e)
    }

    fn unary(&mut self) -> Res<Expr> {
        if self.eat_punct("-") {
            if let Some(Tok::Int(value, suffix)) = self.peek().cloned() {
                if suffix != "i32" || value > i32::MAX as u64 {
                    return Err(format!("negated literal {value}{suffix}"));
                }
                self.pos += 1;
                return Ok(Expr::i32(-(value as i32)));
            }
            let e = self.unary()?;
            return Ok(Expr::Un {
                op: UnOp::Neg,
                ty: Ty::I32,
                e: Box::new(e),
            });
        }
        if self.eat_punct("!") {
            return Ok(Expr::inverted(self.unary()?));
        }
        self.atom()
    }

    fn atom(&mut self) -> Res<Expr> {
        match self.next()? {
            Tok::Int(value, suffix) => match suffix.as_str() {
                "u32" if value <= u64::from(u32::MAX) => Ok(Expr::u32(value as u32)),
                "i32" if value <= i32::MAX as u64 => Ok(Expr::i32(value as i32)),
                _ => Err(format!("literal {value}{suffix}")),
            },
            Tok::Punct("(") => {
                let e = self.expr(0)?;
                self.expect_punct(")")?;
                Ok(e)
            }
            Tok::Ident(name) => match name.as_str() {
                "true" => Ok(Expr::bool(true)),
                "false" => Ok(Expr::bool(false)),
                "i" => Ok(Expr::Gid),
                _ => {
                    if let Some(buf) = BUFFER_NAMES.iter().position(|n| *n == name) {
                        let ty = *self
                            .inputs
                            .get(buf)
                            .ok_or_else(|| format!("unknown buffer {name}"))?;
                        if self.eat_punct("[") {
                            let index = self.expr(0)?;
                            self.expect_punct("]")?;
                            return Ok(Expr::input(buf, ty, index));
                        }
                        self.expect_punct(".")?;
                        if !self.eat_ident("len") {
                            return Err("expected `len`".into());
                        }
                        self.expect_punct("(")?;
                        self.expect_punct(")")?;
                        return Ok(Expr::Len(buf));
                    }
                    let id = Self::var_id(&name).ok_or_else(|| format!("unknown name {name}"))?;
                    let ty = *self
                        .vars
                        .get(&id)
                        .ok_or_else(|| format!("{name} used before its declaration"))?;
                    Ok(Expr::Var(id, ty))
                }
            },
            other => Err(format!("unexpected token {other:?}")),
        }
    }

    fn block(&mut self) -> Res<Vec<Stmt>> {
        self.expect_punct("{")?;
        let mut stmts = Vec::new();
        while !self.eat_punct("}") {
            stmts.push(self.stmt()?);
        }
        Ok(stmts)
    }

    fn stmt(&mut self) -> Res<Stmt> {
        if self.eat_ident("let") {
            let mutable = self.eat_ident("mut");
            let name = self.ident()?;
            let var = Self::var_id(&name).ok_or_else(|| format!("bad variable {name}"))?;
            self.expect_punct(":")?;
            let ty = self.ty()?;
            self.expect_punct("=")?;
            let init = self.expr(0)?;
            self.expect_punct(";")?;
            self.vars.insert(var, ty);
            return Ok(Stmt::Let {
                var,
                ty,
                mutable,
                init,
            });
        }
        if self.eat_ident("if") {
            let cond = self.expr(0)?;
            let then = self.block()?;
            let els = if self.eat_ident("else") {
                self.block()?
            } else {
                Vec::new()
            };
            return Ok(Stmt::If { cond, then, els });
        }
        if self.eat_ident("for") {
            let name = self.ident()?;
            let var = Self::var_id(&name).ok_or_else(|| format!("bad variable {name}"))?;
            if !self.eat_ident("in") {
                return Err("expected `in`".into());
            }
            if self.next()? != Tok::Int(0, "u32".into()) {
                return Err("loop does not start at 0u32".into());
            }
            self.expect_punct("..")?;
            let trips = match self.next()? {
                Tok::Int(n, suffix) if suffix == "u32" => n as u32,
                other => return Err(format!("loop bound {other:?}")),
            };
            self.vars.insert(var, Ty::U32);
            let body = self.block()?;
            return Ok(Stmt::For { var, trips, body });
        }
        let name = self.ident()?;
        let var = if name == "out" {
            self.expect_punct("[")?;
            if !self.eat_ident("i") {
                return Err("store index is not `i`".into());
            }
            self.expect_punct("]")?;
            OUT
        } else {
            Self::var_id(&name).ok_or_else(|| format!("bad statement start {name}"))?
        };
        self.expect_punct("=")?;
        let value = self.expr(0)?;
        self.expect_punct(";")?;
        Ok(Stmt::Assign { var, value })
    }
}

/// Parses the kernel body of an emitted module back into `(body, result)`; `inputs` supplies the buffer types.
pub fn reparse(module_text: &str, inputs: &[InputBuf]) -> Res<(Vec<Stmt>, Expr)> {
    let toks = tokenize(module_text)?;
    let guard: Vec<Tok> = [
        Tok::Ident("if".into()),
        Tok::Ident("i".into()),
        Tok::Punct("<"),
        Tok::Ident("out".into()),
        Tok::Punct("."),
        Tok::Ident("len".into()),
        Tok::Punct("("),
        Tok::Punct(")"),
        Tok::Punct("{"),
    ]
    .into();
    let start = toks
        .windows(guard.len())
        .position(|window| window == guard.as_slice())
        .ok_or("the `if i < out.len()` guard is missing")?;
    let mut parser = Parser {
        toks,
        pos: start + guard.len() - 1,
        vars: BTreeMap::new(),
        inputs: inputs.iter().map(|input| input.ty).collect(),
    };
    let mut body = parser.block()?;
    if parser.toks[parser.pos..]
        .iter()
        .any(|tok| *tok != Tok::Punct("}"))
    {
        return Err("text after the guarded block".into());
    }
    let result = match body.pop() {
        Some(Stmt::Assign { var: OUT, value }) => value,
        Some(Stmt::If { cond, then, els })
            if then
                == [Stmt::Assign {
                    var: OUT,
                    value: Expr::u32(1),
                }]
                && els
                    == [Stmt::Assign {
                        var: OUT,
                        value: Expr::u32(0),
                    }] =>
        {
            cond
        }
        other => return Err(format!("the block does not end in the store: {other:?}")),
    };
    Ok((body, result))
}
