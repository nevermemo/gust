//! A small JSON reader and writer, std only (the tool depends on nothing but `gust`). The reader is the one in
//! `xtask/src/json.rs` (all of RFC 8259; numbers keep their source text) with typed accessors added; the writer
//! prints the files this tool owns: two-space indentation, object keys sorted, LF line ends and one final newline, so
//! the same data always gives the same bytes.

use std::fmt::Write as _;

/// A JSON value. Numbers keep their source text.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Json {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<Json>),
    /// Members in source order; a repeated key keeps every occurrence ([`Json::field`] rejects a repeat).
    Object(Vec<(String, Json)>),
}

/// Deeper nesting is rejected instead of overflowing the stack.
const MAX_DEPTH: usize = 128;

impl Json {
    /// A string value.
    pub(crate) fn text(text: &str) -> Self {
        Self::String(text.to_owned())
    }

    /// A non-negative integer.
    pub(crate) fn number(value: u64) -> Self {
        Self::Number(value.to_string())
    }

    /// An object from members in any order; [`Json::to_pretty`] sorts them by key.
    pub(crate) fn object(members: Vec<(&str, Json)>) -> Self {
        Self::Object(
            members
                .into_iter()
                .map(|(key, value)| (key.to_owned(), value))
                .collect(),
        )
    }

    /// The member `key` of an object (the last one if repeated); `None` for other values.
    pub(crate) fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Self::Object(members) => members
                .iter()
                .rev()
                .find_map(|(name, value)| (name == key).then_some(value)),
            _ => None,
        }
    }

    /// The member `key` of an object, which must exist exactly once. `context` names the value for the error.
    pub(crate) fn field(&self, key: &str, context: &str) -> Result<&Json, String> {
        let Self::Object(members) = self else {
            return Err(format!("{context}: expected an object"));
        };
        let mut found = members.iter().filter(|(name, _)| name == key);
        match (found.next(), found.next()) {
            (Some((_, value)), None) => Ok(value),
            (None, _) => Err(format!("{context}: missing `{key}`")),
            (Some(_), Some(_)) => Err(format!("{context}: `{key}` appears more than once")),
        }
    }

    /// Every key of an object, in source order; empty for other values.
    pub(crate) fn keys(&self) -> Vec<&str> {
        match self {
            Self::Object(members) => members.iter().map(|(name, _)| name.as_str()).collect(),
            _ => Vec::new(),
        }
    }

    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(text) => Some(text),
            _ => None,
        }
    }

    pub(crate) fn as_array(&self) -> Option<&[Json]> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }

    pub(crate) fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }

    /// An integer in `0..=u32::MAX` written as plain digits (no sign, fraction or exponent).
    pub(crate) fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Number(text) if text.bytes().all(|byte| byte.is_ascii_digit()) => {
                text.parse().ok()
            }
            _ => None,
        }
    }

    pub(crate) fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// The text this tool writes for the value, ending in one newline.
    pub(crate) fn to_pretty(&self) -> String {
        let mut text = String::new();
        self.write(&mut text, 0);
        text.push('\n');
        text
    }

    fn write(&self, out: &mut String, indent: usize) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
            Self::Number(number) => out.push_str(number),
            Self::String(text) => write_string(out, text),
            Self::Array(items) if items.is_empty() => out.push_str("[]"),
            // Numbers stay on one line, as in the kernel records.
            Self::Array(items) if items.iter().all(|item| matches!(item, Self::Number(_))) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    item.write(out, indent);
                }
                out.push(']');
            }
            Self::Array(items) => {
                out.push_str("[\n");
                for (index, item) in items.iter().enumerate() {
                    out.push_str(&"  ".repeat(indent + 1));
                    item.write(out, indent + 1);
                    out.push_str(if index + 1 < items.len() { ",\n" } else { "\n" });
                }
                out.push_str(&"  ".repeat(indent));
                out.push(']');
            }
            Self::Object(members) if members.is_empty() => out.push_str("{}"),
            Self::Object(members) => {
                let mut sorted: Vec<&(String, Json)> = members.iter().collect();
                sorted.sort_by(|a, b| a.0.cmp(&b.0));
                out.push_str("{\n");
                for (index, (key, value)) in sorted
                    .iter()
                    .map(|member| (&member.0, &member.1))
                    .enumerate()
                {
                    out.push_str(&"  ".repeat(indent + 1));
                    write_string(out, key);
                    out.push_str(": ");
                    value.write(out, indent + 1);
                    out.push_str(if index + 1 < sorted.len() {
                        ",\n"
                    } else {
                        "\n"
                    });
                }
                out.push_str(&"  ".repeat(indent));
                out.push('}');
            }
        }
    }
}

/// A JSON string literal: `"`, `\` and control characters escaped, everything else as UTF-8.
fn write_string(out: &mut String, text: &str) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Parses one JSON document; trailing text other than whitespace is an error.
pub(crate) fn parse(text: &str) -> Result<Json, String> {
    let mut parser = Parser {
        bytes: text.as_bytes(),
        text,
        at: 0,
    };
    let value = parser.value(0)?;
    parser.skip_space();
    if parser.at != parser.bytes.len() {
        return Err(parser.error("unexpected text after the JSON value"));
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    text: &'a str,
    at: usize,
}

impl Parser<'_> {
    fn error(&self, message: &str) -> String {
        format!("invalid JSON at byte {}: {message}", self.at)
    }

    fn skip_space(&mut self) {
        while matches!(self.bytes.get(self.at), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.at += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }

    fn expect(&mut self, wanted: u8) -> Result<(), String> {
        if self.peek() == Some(wanted) {
            self.at += 1;
            Ok(())
        } else {
            Err(self.error(&format!("expected `{}`", wanted as char)))
        }
    }

    fn literal(&mut self, word: &str, value: Json) -> Result<Json, String> {
        if self.bytes[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(value)
        } else {
            Err(self.error("unknown literal"))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, String> {
        if depth > MAX_DEPTH {
            return Err(self.error("nesting is too deep"));
        }
        self.skip_space();
        match self.peek() {
            None => Err(self.error("unexpected end of input")),
            Some(b'n') => self.literal("null", Json::Null),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'"') => self.string().map(Json::String),
            Some(b'[') => self.array(depth),
            Some(b'{') => self.object(depth),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => Err(self.error("unexpected character")),
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json, String> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_space();
        if self.peek() == Some(b']') {
            self.at += 1;
            return Ok(Json::Array(items));
        }
        loop {
            items.push(self.value(depth + 1)?);
            self.skip_space();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(Json::Array(items));
                }
                _ => return Err(self.error("expected `,` or `]`")),
            }
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, String> {
        self.expect(b'{')?;
        let mut members = Vec::new();
        self.skip_space();
        if self.peek() == Some(b'}') {
            self.at += 1;
            return Ok(Json::Object(members));
        }
        loop {
            self.skip_space();
            let key = self.string()?;
            self.skip_space();
            self.expect(b':')?;
            members.push((key, self.value(depth + 1)?));
            self.skip_space();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Json::Object(members));
                }
                _ => return Err(self.error("expected `,` or `}`")),
            }
        }
    }

    /// `-? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE] [+-]? [0-9]+)?`, kept as written.
    fn number(&mut self) -> Result<Json, String> {
        let start = self.at;
        if self.peek() == Some(b'-') {
            self.at += 1;
        }
        match self.peek() {
            Some(b'0') => self.at += 1,
            Some(b'1'..=b'9') => self.digits(),
            _ => return Err(self.error("a number needs a digit")),
        }
        if self.peek() == Some(b'.') {
            self.at += 1;
            if !self.peek().is_some_and(|b| b.is_ascii_digit()) {
                return Err(self.error("a fraction needs a digit"));
            }
            self.digits();
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.at += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.at += 1;
            }
            if !self.peek().is_some_and(|b| b.is_ascii_digit()) {
                return Err(self.error("an exponent needs a digit"));
            }
            self.digits();
        }
        Ok(Json::Number(self.text[start..self.at].to_owned()))
    }

    fn digits(&mut self) {
        while self.peek().is_some_and(|b| b.is_ascii_digit()) {
            self.at += 1;
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            // Copy the run of ordinary characters in one step; `"` and `\` are ASCII, so the
            // run always ends on a character boundary.
            let start = self.at;
            while let Some(byte) = self.peek() {
                if byte == b'"' || byte == b'\\' || byte < 0x20 {
                    break;
                }
                self.at += 1;
            }
            out.push_str(&self.text[start..self.at]);
            match self.peek() {
                None => return Err(self.error("unterminated string")),
                Some(b'"') => {
                    self.at += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.at += 1;
                    out.push(self.escape()?);
                }
                Some(_) => return Err(self.error("control character in string")),
            }
        }
    }

    /// The character after a backslash.
    fn escape(&mut self) -> Result<char, String> {
        let Some(byte) = self.peek() else {
            return Err(self.error("unterminated escape"));
        };
        self.at += 1;
        Ok(match byte {
            b'"' => '"',
            b'\\' => '\\',
            b'/' => '/',
            b'b' => '\u{8}',
            b'f' => '\u{c}',
            b'n' => '\n',
            b'r' => '\r',
            b't' => '\t',
            b'u' => {
                let first = self.hex4()?;
                if (0xD800..0xDC00).contains(&first) {
                    // A high surrogate must be followed by `\uDC00..DFFF`.
                    if !self.bytes[self.at..].starts_with(b"\\u") {
                        return Err(self.error("lone high surrogate"));
                    }
                    self.at += 2;
                    let second = self.hex4()?;
                    if !(0xDC00..0xE000).contains(&second) {
                        return Err(self.error("high surrogate without a low surrogate"));
                    }
                    let code = 0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00);
                    char::from_u32(code).ok_or_else(|| self.error("invalid surrogate pair"))?
                } else {
                    char::from_u32(first).ok_or_else(|| self.error("lone low surrogate"))?
                }
            }
            _ => return Err(self.error("unknown escape")),
        })
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let digits = self
            .bytes
            .get(self.at..self.at + 4)
            .filter(|digits| digits.iter().all(u8::is_ascii_hexdigit))
            .ok_or_else(|| self.error("`\\u` needs four hex digits"))?;
        // The slice is four ASCII hex digits, so it is valid UTF-8 and parses.
        let value = u32::from_str_radix(std::str::from_utf8(digits).unwrap_or("0"), 16)
            .map_err(|_| self.error("`\\u` needs four hex digits"))?;
        self.at += 4;
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::{Json, parse};

    #[test]
    fn the_reader_accepts_rfc_8259_and_rejects_the_usual_mistakes() {
        let value =
            parse(r#"{"a": [1, 2.5e3, -0, true, null], "b": "x\u00e9\n\ud83d\ude80"}"#).unwrap();
        assert_eq!(value.get("b").unwrap().as_str(), Some("x\u{e9}\n\u{1f680}"));
        assert_eq!(value.get("a").unwrap().as_array().unwrap().len(), 5);
        for bad in [
            "",
            "{",
            "[1,]",
            "{\"a\":1,}",
            "01",
            "1.",
            "\"\\ud800\"",
            "\"tab\there\"",
            "{} {}",
            "nul",
        ] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
        let deep = format!("{}0{}", "[".repeat(200), "]".repeat(200));
        assert!(parse(&deep).unwrap_err().contains("too deep"));
    }

    #[test]
    fn field_rejects_a_missing_and_a_repeated_key_and_as_u32_rejects_signs_and_fractions() {
        let value =
            parse(r#"{"x": 1, "y": 2, "y": 3, "n": -1, "f": 1.0, "big": 4294967296}"#).unwrap();
        assert_eq!(value.field("x", "v").unwrap().as_u32(), Some(1));
        assert_eq!(value.field("z", "v").unwrap_err(), "v: missing `z`");
        assert_eq!(
            value.field("y", "v").unwrap_err(),
            "v: `y` appears more than once"
        );
        for key in ["n", "f", "big"] {
            assert_eq!(value.field(key, "v").unwrap().as_u32(), None, "{key}");
        }
        assert_eq!(
            Json::Null.field("x", "v").unwrap_err(),
            "v: expected an object"
        );
    }

    #[test]
    fn the_writer_sorts_keys_at_every_level_and_is_deterministic() {
        let value = Json::object(vec![
            ("zeta", Json::number(1)),
            (
                "alpha",
                Json::object(vec![("b", Json::Bool(true)), ("a", Json::Null)]),
            ),
            ("list", Json::Array(vec![Json::number(3), Json::number(4)])),
            ("words", Json::Array(vec![Json::text("q\"\\\n\u{1}")])),
            ("empty", Json::Array(vec![])),
        ]);
        let expected = "{\n  \"alpha\": {\n    \"a\": null,\n    \"b\": true\n  },\n  \"empty\": [],\n  \"list\": [3, 4],\n  \"words\": [\n    \"q\\\"\\\\\\n\\u0001\"\n  ],\n  \"zeta\": 1\n}\n";
        assert_eq!(value.to_pretty(), expected);
        assert_eq!(parse(expected).unwrap().to_pretty(), expected);
        assert!(!expected.contains('\r'));
    }
}
