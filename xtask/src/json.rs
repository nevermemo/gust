//! A small JSON reader for the output of `cargo metadata`. xtask depends on nothing, so there
//! is no serde; this reads the whole of RFC 8259 and keeps only what the callers look at.

/// A parsed JSON value. Numbers keep their source text (nothing here needs arithmetic).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Json {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<Json>),
    /// Members in source order; a repeated key keeps every occurrence, `get` finds the last.
    Object(Vec<(String, Json)>),
}

/// Deeper nesting is rejected instead of overflowing the stack.
const MAX_DEPTH: usize = 128;

impl Json {
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

    /// The string, or `None` for JSON `null` and for every non-string.
    pub(crate) fn string_or_null(&self) -> Option<String> {
        self.as_str().map(str::to_owned)
    }
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

/// `value` as a JSON string literal: quoted, with the escapes of RFC 8259.
pub(crate) fn json_string(value: &str) -> String {
    let mut out = String::from("\"");
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The typed accessors of a difftest `report.json`: `field` is the member `key` of an object, and each of the others
/// reads one type from it. A missing or mistyped member is an error that names `report.json`.
pub(crate) fn field<'a>(json: &'a Json, key: &str) -> Result<&'a Json, String> {
    json.get(key)
        .ok_or_else(|| format!("report.json has no `{key}`"))
}

pub(crate) fn uint(json: &Json, key: &str) -> Result<u64, String> {
    match field(json, key)? {
        Json::Number(text) => text
            .parse()
            .map_err(|_| format!("report.json `{key}` is not a whole number: {text}")),
        _ => Err(format!("report.json `{key}` is not a number")),
    }
}

pub(crate) fn float(json: &Json, key: &str) -> Result<f64, String> {
    match field(json, key)? {
        Json::Number(text) => text
            .parse()
            .map_err(|_| format!("report.json `{key}` is not a number: {text}")),
        _ => Err(format!("report.json `{key}` is not a number")),
    }
}

pub(crate) fn text(json: &Json, key: &str) -> Result<String, String> {
    field(json, key)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("report.json `{key}` is not a string"))
}

pub(crate) fn array<'a>(json: &'a Json, key: &str) -> Result<&'a [Json], String> {
    field(json, key)?
        .as_array()
        .ok_or_else(|| format!("report.json `{key}` is not an array"))
}

#[cfg(test)]
#[path = "tests/json.rs"]
mod json_tests;
