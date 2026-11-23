use super::error::{Error, invalid};
use std::collections::BTreeMap;

#[derive(Debug)]
pub(super) enum Json {
    Object(BTreeMap<String, Json>),
    Array(Vec<Json>),
    String(String),
    Number(String),
    Other,
}
impl Json {
    pub(super) fn optional(&self, key: &str) -> Result<Option<&Self>, Error> {
        match self {
            Self::Object(m) => Ok(m.get(key)),
            _ => Err(invalid("expected object")),
        }
    }
    pub(super) fn get(&self, key: &str) -> Result<&Self, Error> {
        self.optional(key)?
            .ok_or_else(|| invalid(format!("missing {key}")))
    }
    pub(super) fn array(&self) -> Result<&[Self], Error> {
        match self {
            Self::Array(a) => Ok(a),
            _ => Err(invalid("expected array")),
        }
    }
    pub(super) fn string(&self) -> Result<&str, Error> {
        match self {
            Self::String(s) => Ok(s),
            _ => Err(invalid("expected string")),
        }
    }
    pub(super) fn u32(&self) -> Result<u32, Error> {
        match self {
            Self::Number(s) if s.bytes().all(|b| b.is_ascii_digit()) => {
                s.parse().map_err(|_| invalid("integer exceeds u32"))
            }
            _ => Err(invalid("expected unsigned integer")),
        }
    }
}
pub(super) struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}
impl<'a> Parser<'a> {
    pub(super) fn parse(source: &'a str) -> Result<Json, Error> {
        let mut p = Self {
            bytes: source.as_bytes(),
            pos: 0,
        };
        let value = p.value(0)?;
        p.ws();
        if p.pos != p.bytes.len() {
            return Err(invalid("trailing JSON data"));
        }
        Ok(value)
    }
    fn ws(&mut self) {
        while self
            .bytes
            .get(self.pos)
            .is_some_and(|b| matches!(b, b' ' | b'\r' | b'\n' | b'\t'))
        {
            self.pos += 1;
        }
    }
    fn take(&mut self, b: u8) -> bool {
        self.ws();
        if self.bytes.get(self.pos) == Some(&b) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, b: u8) -> Result<(), Error> {
        if self.take(b) {
            Ok(())
        } else {
            Err(invalid(format!(
                "expected {} at byte {}",
                b as char, self.pos
            )))
        }
    }
    fn value(&mut self, depth: usize) -> Result<Json, Error> {
        if depth > 64 {
            return Err(invalid("JSON nesting exceeds 64"));
        }
        self.ws();
        match self.bytes.get(self.pos).copied() {
            Some(b'{') => {
                self.pos += 1;
                let mut fields = BTreeMap::new();
                if !self.take(b'}') {
                    loop {
                        self.ws();
                        let key = self.string()?;
                        self.expect(b':')?;
                        let value = self.value(depth + 1)?;
                        if fields.insert(key, value).is_some() {
                            return Err(invalid("duplicate JSON key"));
                        }
                        if self.take(b'}') {
                            break;
                        }
                        self.expect(b',')?;
                    }
                }
                Ok(Json::Object(fields))
            }
            Some(b'[') => {
                self.pos += 1;
                let mut values = Vec::new();
                if !self.take(b']') {
                    loop {
                        values.push(self.value(depth + 1)?);
                        if self.take(b']') {
                            break;
                        }
                        self.expect(b',')?;
                    }
                }
                Ok(Json::Array(values))
            }
            Some(b'"') => self.string().map(Json::String),
            Some(b'-' | b'0'..=b'9') => self.number().map(Json::Number),
            _ => {
                for literal in [b"true".as_slice(), b"false", b"null"] {
                    if self.bytes[self.pos..].starts_with(literal) {
                        self.pos += literal.len();
                        return Ok(Json::Other);
                    }
                }
                Err(invalid(format!("invalid JSON at byte {}", self.pos)))
            }
        }
    }
    fn number(&mut self) -> Result<String, Error> {
        let start = self.pos;
        if self.bytes.get(self.pos) == Some(&b'-') {
            self.pos += 1;
        }
        if self.bytes.get(self.pos) == Some(&b'0') {
            self.pos += 1;
        } else {
            self.digits()?;
        }
        if self.bytes.get(self.pos) == Some(&b'.') {
            self.pos += 1;
            self.digits()?;
        }
        if self
            .bytes
            .get(self.pos)
            .is_some_and(|b| matches!(b, b'e' | b'E'))
        {
            self.pos += 1;
            if self
                .bytes
                .get(self.pos)
                .is_some_and(|b| matches!(b, b'+' | b'-'))
            {
                self.pos += 1;
            }
            self.digits()?;
        }
        Ok(String::from_utf8(self.bytes[start..self.pos].to_vec()).expect("ASCII JSON number"))
    }
    fn digits(&mut self) -> Result<(), Error> {
        let start = self.pos;
        while self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) {
            self.pos += 1;
        }
        if self.pos == start {
            Err(invalid("expected digit"))
        } else {
            Ok(())
        }
    }
    fn hex(&mut self) -> Result<u16, Error> {
        let mut n = 0;
        for _ in 0..4 {
            let b = *self
                .bytes
                .get(self.pos)
                .ok_or_else(|| invalid("truncated Unicode escape"))?;
            self.pos += 1;
            n = n * 16
                + (b as char)
                    .to_digit(16)
                    .ok_or_else(|| invalid("invalid Unicode escape"))? as u16;
        }
        Ok(n)
    }
    fn string(&mut self) -> Result<String, Error> {
        self.expect(b'"')?;
        let mut result = String::new();
        let mut start = self.pos;
        loop {
            let b = *self
                .bytes
                .get(self.pos)
                .ok_or_else(|| invalid("unterminated string"))?;
            if b == b'"' || b == b'\\' {
                result.push_str(
                    std::str::from_utf8(&self.bytes[start..self.pos])
                        .map_err(|_| invalid("invalid UTF-8"))?,
                );
                self.pos += 1;
                if b == b'"' {
                    return Ok(result);
                }
                let escape = *self
                    .bytes
                    .get(self.pos)
                    .ok_or_else(|| invalid("truncated escape"))?;
                self.pos += 1;
                result.push(match escape {
                    b'"' => '"',
                    b'\\' => '\\',
                    b'/' => '/',
                    b'b' => '\u{8}',
                    b'f' => '\u{c}',
                    b'n' => '\n',
                    b'r' => '\r',
                    b't' => '\t',
                    b'u' => {
                        let high = self.hex()?;
                        let cp = if (0xd800..=0xdbff).contains(&high) {
                            if self.bytes.get(self.pos..self.pos + 2) != Some(b"\\u") {
                                return Err(invalid("missing low surrogate"));
                            }
                            self.pos += 2;
                            let low = self.hex()?;
                            if !(0xdc00..=0xdfff).contains(&low) {
                                return Err(invalid("invalid low surrogate"));
                            }
                            0x10000 + ((high as u32 - 0xd800) << 10) + (low as u32 - 0xdc00)
                        } else {
                            high as u32
                        };
                        char::from_u32(cp).ok_or_else(|| invalid("invalid Unicode scalar"))?
                    }
                    _ => return Err(invalid("invalid escape")),
                });
                start = self.pos;
            } else {
                if b < 0x20 {
                    return Err(invalid("control byte in string"));
                }
                self.pos += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_json_syntax_and_unicode() {
        let json =
            Parser::parse(r#"{"name":"a\uD83D\uDE80é\n","unknown":[true,false,null,-1.25e+2]}"#)
                .unwrap();
        assert_eq!(json.get("name").unwrap().string().unwrap(), "a🚀é\n");
        for source in [
            "01",
            "-",
            "1.",
            "1e+",
            "[1,]",
            r#""\uD800""#,
            r#""\uDC00""#,
            r#""\uD800\u0041""#,
            r#""\q""#,
            "\"raw\nline\"",
            "{\"x\":0,\"x\":1}",
            "true false",
        ] {
            assert!(Parser::parse(source).is_err(), "{source}");
        }
        let deep = format!("{}0{}", "[".repeat(66), "]".repeat(66));
        assert!(Parser::parse(&deep).is_err());
        for source in ["4294967296", "-1", "1.0", "1e1"] {
            assert!(Parser::parse(source).unwrap().u32().is_err());
        }
    }
}
