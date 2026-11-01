//! A small JSON value, parser and writer (std only). The generated tests write one JSON file per program and the
//! harness writes `report.json`; both are plain data, so a 150-line parser is the whole dependency.
//!
//! Numbers are `Int` (written without a fraction or exponent) or `Float`. Objects keep their insertion order, so a
//! report is the same bytes for the same data. Strings are escaped for `"`, `\`, and control characters; the parser
//! reads the same escapes plus `\/`, `\b`, `\f` and `\uXXXX` for the basic plane.

use std::fmt::Write as _;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn obj(fields: Vec<(&str, Json)>) -> Json {
        Json::Obj(
            fields
                .into_iter()
                .map(|(key, value)| (key.to_owned(), value))
                .collect(),
        )
    }

    pub fn str(text: &str) -> Json {
        Json::Str(text.to_owned())
    }

    pub fn uint(value: u64) -> Json {
        Json::Int(i64::try_from(value).unwrap_or(i64::MAX))
    }

    pub fn words(words: &[u32]) -> Json {
        Json::Arr(words.iter().map(|w| Json::Int(i64::from(*w))).collect())
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(text) => Some(text),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Int(value) => u64::try_from(*value).ok(),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(items) => Some(items),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }

    /// An array of `u32` values, or `None` if this is not one.
    pub fn as_words(&self) -> Option<Vec<u32>> {
        self.as_array()?
            .iter()
            .map(|item| item.as_u64().and_then(|v| u32::try_from(v).ok()))
            .collect()
    }

    /// Pretty text, two spaces per level; an array of numbers stays on one line.
    pub fn pretty(&self) -> String {
        let mut out = String::new();
        self.write(&mut out, 0);
        out.push('\n');
        out
    }

    fn is_flat(&self) -> bool {
        match self {
            Json::Arr(items) => items
                .iter()
                .all(|item| !matches!(item, Json::Arr(_) | Json::Obj(_))),
            _ => false,
        }
    }

    fn write(&self, out: &mut String, level: usize) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
            Json::Int(value) => {
                let _ = write!(out, "{value}");
            }
            Json::Float(value) => {
                if value.is_finite() {
                    let _ = write!(out, "{value:?}");
                } else {
                    out.push_str("null");
                }
            }
            Json::Str(text) => escape(text, out),
            Json::Arr(items) if items.is_empty() => out.push_str("[]"),
            Json::Arr(items) if self.is_flat() => {
                out.push('[');
                for (n, item) in items.iter().enumerate() {
                    if n > 0 {
                        out.push_str(", ");
                    }
                    item.write(out, level);
                }
                out.push(']');
            }
            Json::Arr(items) => {
                out.push_str("[\n");
                for (n, item) in items.iter().enumerate() {
                    out.push_str(&"  ".repeat(level + 1));
                    item.write(out, level + 1);
                    out.push_str(if n + 1 < items.len() { ",\n" } else { "\n" });
                }
                out.push_str(&"  ".repeat(level));
                out.push(']');
            }
            Json::Obj(fields) if fields.is_empty() => out.push_str("{}"),
            Json::Obj(fields) => {
                out.push_str("{\n");
                for (n, (key, value)) in fields.iter().enumerate() {
                    out.push_str(&"  ".repeat(level + 1));
                    escape(key, out);
                    out.push_str(": ");
                    value.write(out, level + 1);
                    out.push_str(if n + 1 < fields.len() { ",\n" } else { "\n" });
                }
                out.push_str(&"  ".repeat(level));
                out.push('}');
            }
        }
    }
}

fn escape(text: &str, out: &mut String) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Parses one JSON document (trailing whitespace allowed, nothing else).
pub fn parse(text: &str) -> Result<Json, String> {
    let mut parser = Parser {
        text,
        bytes: text.as_bytes(),
        pos: 0,
    };
    let value = parser.value(0)?;
    parser.skip_space();
    if parser.pos != parser.bytes.len() {
        return Err(format!("trailing text at byte {}", parser.pos));
    }
    Ok(value)
}

struct Parser<'a> {
    text: &'a str,
    bytes: &'a [u8],
    pos: usize,
}

const MAX_DEPTH: usize = 64;

impl Parser<'_> {
    fn skip_space(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn expect(&mut self, byte: u8) -> Result<(), String> {
        if self.peek() == Some(byte) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!("expected {:?} at byte {}", byte as char, self.pos))
        }
    }

    fn literal(&mut self, word: &str, value: Json) -> Result<Json, String> {
        if self.bytes[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
            Ok(value)
        } else {
            Err(format!("bad literal at byte {}", self.pos))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, String> {
        if depth > MAX_DEPTH {
            return Err("nesting too deep".into());
        }
        self.skip_space();
        match self.peek() {
            None => Err("unexpected end of text".into()),
            Some(b'n') => self.literal("null", Json::Null),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'"') => self.string().map(Json::Str),
            Some(b'[') => {
                self.pos += 1;
                let mut items = Vec::new();
                self.skip_space();
                if self.peek() == Some(b']') {
                    self.pos += 1;
                    return Ok(Json::Arr(items));
                }
                loop {
                    items.push(self.value(depth + 1)?);
                    self.skip_space();
                    match self.peek() {
                        Some(b',') => self.pos += 1,
                        Some(b']') => {
                            self.pos += 1;
                            return Ok(Json::Arr(items));
                        }
                        _ => return Err(format!("expected , or ] at byte {}", self.pos)),
                    }
                }
            }
            Some(b'{') => {
                self.pos += 1;
                let mut fields = Vec::new();
                self.skip_space();
                if self.peek() == Some(b'}') {
                    self.pos += 1;
                    return Ok(Json::Obj(fields));
                }
                loop {
                    self.skip_space();
                    let key = self.string()?;
                    self.skip_space();
                    self.expect(b':')?;
                    fields.push((key, self.value(depth + 1)?));
                    self.skip_space();
                    match self.peek() {
                        Some(b',') => self.pos += 1,
                        Some(b'}') => {
                            self.pos += 1;
                            return Ok(Json::Obj(fields));
                        }
                        _ => return Err(format!("expected , or }} at byte {}", self.pos)),
                    }
                }
            }
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(other) => Err(format!(
                "unexpected {:?} at byte {}",
                other as char, self.pos
            )),
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.pos;
        while self
            .peek()
            .is_some_and(|b| b.is_ascii_digit() || matches!(b, b'-' | b'+' | b'.' | b'e' | b'E'))
        {
            self.pos += 1;
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|e| e.to_string())?;
        if text.contains(['.', 'e', 'E']) {
            text.parse::<f64>()
                .map(Json::Float)
                .map_err(|e| format!("number {text:?}: {e}"))
        } else {
            text.parse::<i64>()
                .map(Json::Int)
                .map_err(|e| format!("number {text:?}: {e}"))
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let rest = &self.bytes[self.pos..];
            let Some(&byte) = rest.first() else {
                return Err("unterminated string".into());
            };
            match byte {
                b'"' => {
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    let Some(&escape) = rest.get(1) else {
                        return Err("unterminated escape".into());
                    };
                    self.pos += 2;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'u' => {
                            let hex = self
                                .bytes
                                .get(self.pos..self.pos + 4)
                                .and_then(|h| std::str::from_utf8(h).ok())
                                .ok_or("short \\u escape")?;
                            let code = u32::from_str_radix(hex, 16).map_err(|e| e.to_string())?;
                            out.push(char::from_u32(code).unwrap_or('\u{fffd}'));
                            self.pos += 4;
                        }
                        other => return Err(format!("bad escape \\{}", other as char)),
                    }
                }
                _ => {
                    // Copy one UTF-8 scalar (`pos` is always on a character boundary).
                    let ch = self.text[self.pos..]
                        .chars()
                        .next()
                        .ok_or("unterminated string")?;
                    out.push(ch);
                    self.pos += ch.len_utf8();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_round_trip_through_text() {
        let value = Json::obj(vec![
            ("name", Json::str("a \"quoted\"\nline\t\\")),
            ("words", Json::words(&[0, 1, u32::MAX])),
            ("none", Json::Null),
            ("flag", Json::Bool(true)),
            ("ratio", Json::Float(0.1)),
            (
                "nested",
                Json::Arr(vec![
                    Json::obj(vec![("k", Json::Int(-3))]),
                    Json::Arr(vec![]),
                ]),
            ),
            ("empty", Json::Obj(vec![])),
        ]);
        let text = value.pretty();
        assert_eq!(parse(&text), Ok(value));
    }

    #[test]
    fn parsing_rejects_malformed_text() {
        for bad in ["", "{", "[1,]", "{\"a\" 1}", "tru", "\"abc", "1 2", "[1 2]"] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
        assert_eq!(parse(" [ 1 , 2 ] "), Ok(Json::words(&[1, 2])));
        assert_eq!(parse("\"\\u0041\\u00e9\""), Ok(Json::str("A\u{e9}")));
        assert_eq!(parse("\"é\""), Ok(Json::str("\u{e9}")));
    }

    #[test]
    fn words_reject_values_outside_u32() {
        assert_eq!(
            parse("[0, 4294967295]").unwrap().as_words().unwrap().len(),
            2
        );
        assert!(parse("[4294967296]").unwrap().as_words().is_none());
        assert!(parse("[-1]").unwrap().as_words().is_none());
        assert!(parse("[1.5]").unwrap().as_words().is_none());
    }
}
