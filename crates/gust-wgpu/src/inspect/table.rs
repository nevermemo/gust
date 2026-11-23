//! Values, columns and the table: layout flattening, decoding, text and JSON.

use std::fmt::{self, Write as _};

use gust::{ScalarKind, TypeLayout, TypeLayoutKind};

use super::InspectError;

/// One 32-bit field value read from a buffer.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Value {
    /// A `u32` field.
    U32(u32),
    /// An `i32` field.
    I32(i32),
    /// An `f32` field.
    F32(f32),
}

impl Value {
    /// Whether the value is an `f32` NaN.
    pub fn is_nan(self) -> bool {
        matches!(self, Self::F32(value) if value.is_nan())
    }

    /// The order of two values of one kind, a NaN placed by `f32::total_cmp`; values of two kinds are equal.
    pub(super) fn total_cmp(self, other: Self) -> std::cmp::Ordering {
        match (self, other) {
            (Self::U32(a), Self::U32(b)) => a.cmp(&b),
            (Self::I32(a), Self::I32(b)) => a.cmp(&b),
            (Self::F32(a), Self::F32(b)) => a.total_cmp(&b),
            _ => std::cmp::Ordering::Equal,
        }
    }

    /// Whether two values are the same bits: a NaN equals itself, and `0.0` differs from `-0.0`, which is what a
    /// before/after diff of memory wants.
    pub(super) fn same_bits(self, other: Self) -> bool {
        match (self, other) {
            (Self::U32(a), Self::U32(b)) => a == b,
            (Self::I32(a), Self::I32(b)) => a == b,
            (Self::F32(a), Self::F32(b)) => a.to_bits() == b.to_bits(),
            _ => false,
        }
    }

    pub(super) fn write_json(self, out: &mut String) {
        match self {
            Self::U32(value) => {
                let _ = write!(out, "{value}");
            }
            Self::I32(value) => {
                let _ = write!(out, "{value}");
            }
            Self::F32(value) if value.is_finite() => {
                let _ = write!(out, "{value:?}");
            }
            // JSON has no NaN or infinity: they are strings.
            Self::F32(value) => {
                let _ = write!(out, "\"{value:?}\"");
            }
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::U32(value) => fmt::Display::fmt(value, formatter),
            Self::I32(value) => fmt::Display::fmt(value, formatter),
            // Debug keeps `1.0` a float.
            Self::F32(value) => fmt::Debug::fmt(value, formatter),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Kind {
    U32,
    I32,
    F32,
}

#[derive(Clone, Debug)]
pub(super) struct Column {
    pub(super) name: String,
    pub(super) kind: Kind,
    pub(super) offset: usize,
}

fn kind_of(scalar: ScalarKind, name: &'static str) -> Result<Kind, InspectError> {
    match scalar {
        ScalarKind::U32 => Ok(Kind::U32),
        ScalarKind::I32 => Ok(Kind::I32),
        ScalarKind::F32 => Ok(Kind::F32),
        _ => Err(InspectError::Unsupported(name)),
    }
}

/// Flatten `layout`, found `offset` bytes into an element, into one column per scalar.
fn flatten(
    layout: &TypeLayout,
    name: &str,
    offset: u32,
    out: &mut Vec<Column>,
) -> Result<(), InspectError> {
    let mut leaf = |name: String, kind: Kind, offset: u32| {
        out.push(Column {
            name,
            kind,
            offset: offset as usize,
        });
    };
    match layout.kind {
        TypeLayoutKind::Scalar(scalar) => {
            let name = if name.is_empty() { "value" } else { name };
            leaf(name.to_owned(), kind_of(scalar, layout.name)?, offset);
        }
        TypeLayoutKind::Vector { scalar, count } => {
            let kind = kind_of(scalar, layout.name)?;
            for component in 0..count {
                let letter = ["x", "y", "z", "w"][component as usize];
                let name = if name.is_empty() {
                    letter.to_owned()
                } else {
                    format!("{name}.{letter}")
                };
                leaf(name, kind, offset + 4 * component);
            }
        }
        TypeLayoutKind::Matrix {
            scalar,
            rows,
            cols,
            column_stride,
        } => {
            let kind = kind_of(scalar, layout.name)?;
            for column in 0..cols {
                for row in 0..rows {
                    leaf(
                        format!("{name}[{column}][{row}]"),
                        kind,
                        offset + column * column_stride + 4 * row,
                    );
                }
            }
        }
        TypeLayoutKind::Struct(fields) => {
            for field in fields {
                let inner = if name.is_empty() {
                    field.name.to_owned()
                } else {
                    format!("{name}.{}", field.name)
                };
                flatten(&field.ty, &inner, offset + field.offset, out)?;
            }
        }
        TypeLayoutKind::Array { element, count } => {
            for index in 0..count {
                flatten(
                    element,
                    &format!("{name}[{index}]"),
                    offset + index * element.stride,
                    out,
                )?;
            }
        }
        _ => return Err(InspectError::Unsupported(layout.name)),
    }
    Ok(())
}

pub(super) fn columns_of(layout: &TypeLayout) -> Result<Vec<Column>, InspectError> {
    let mut columns = Vec::new();
    flatten(layout, "", 0, &mut columns)?;
    Ok(columns)
}

fn word(bytes: &[u8], at: usize) -> [u8; 4] {
    [bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]
}

/// The rows of a range of elements: every column's value per element, with the first element's index.
#[derive(Clone, Debug)]
pub struct InspectTable {
    type_name: &'static str,
    pub(super) first: usize,
    pub(super) columns: Vec<Column>,
    pub(super) rows: Vec<Vec<Value>>,
}

impl InspectTable {
    pub(super) fn decode(
        layout: &TypeLayout,
        first: usize,
        bytes: &[u8],
        count: usize,
    ) -> Result<Self, InspectError> {
        let columns = columns_of(layout)?;
        let stride = layout.stride as usize;
        let rows = (0..count)
            .map(|element| {
                columns
                    .iter()
                    .map(|column| {
                        let raw = word(bytes, element * stride + column.offset);
                        match column.kind {
                            Kind::U32 => Value::U32(u32::from_le_bytes(raw)),
                            Kind::I32 => Value::I32(i32::from_le_bytes(raw)),
                            Kind::F32 => Value::F32(f32::from_le_bytes(raw)),
                        }
                    })
                    .collect()
            })
            .collect();
        Ok(Self {
            type_name: layout.name,
            first,
            columns,
            rows,
        })
    }

    /// The element type's name.
    pub fn type_name(&self) -> &'static str {
        self.type_name
    }

    /// The buffer index of the first row.
    pub fn first_index(&self) -> usize {
        self.first
    }

    /// The column names, in layout order.
    pub fn columns(&self) -> impl ExactSizeIterator<Item = &str> {
        self.columns.iter().map(|column| column.name.as_str())
    }

    /// The rows: one per element, one value per column.
    pub fn rows(&self) -> &[Vec<Value>] {
        &self.rows
    }

    /// The table as JSON:
    /// `{"type":..,"first":..,"count":..,"fields":[..],"rows":[{"index":..,"values":{"<field>":..}}]}`.
    /// A NaN or an infinity is the string `"NaN"`, `"inf"` or `"-inf"`.
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        out.push_str("{\"type\":");
        json_string(self.type_name, &mut out);
        let _ = write!(
            out,
            ",\"first\":{},\"count\":{},\"fields\":[",
            self.first,
            self.rows.len()
        );
        for (position, column) in self.columns.iter().enumerate() {
            if position > 0 {
                out.push(',');
            }
            json_string(&column.name, &mut out);
        }
        out.push_str("],\"rows\":[");
        for (row, values) in self.rows.iter().enumerate() {
            if row > 0 {
                out.push(',');
            }
            let _ = write!(out, "{{\"index\":{},\"values\":{{", self.first + row);
            for (position, (column, value)) in self.columns.iter().zip(values).enumerate() {
                if position > 0 {
                    out.push(',');
                }
                json_string(&column.name, &mut out);
                out.push(':');
                value.write_json(&mut out);
            }
            out.push_str("}}");
        }
        out.push_str("]}");
        out
    }
}

fn json_string(text: &str, out: &mut String) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            control if control < ' ' => {
                let _ = write!(out, "\\u{:04x}", u32::from(control));
            }
            other => out.push(other),
        }
    }
    out.push('"');
}

/// Right-align `cells` (the header first) into columns joined by ` | `, with a rule under the header.
pub(super) fn write_grid(formatter: &mut fmt::Formatter<'_>, cells: &[Vec<String>]) -> fmt::Result {
    let width = |column: usize| cells.iter().map(|row| row[column].len()).max().unwrap_or(0);
    let widths: Vec<usize> = (0..cells.first().map_or(0, Vec::len)).map(width).collect();
    for (row_number, row) in cells.iter().enumerate() {
        let line: Vec<String> = row
            .iter()
            .zip(&widths)
            .map(|(cell, width)| format!("{cell:>width$}"))
            .collect();
        writeln!(formatter, "{}", line.join(" | "))?;
        if row_number == 0 {
            let rule: Vec<String> = widths.iter().map(|width| "-".repeat(*width)).collect();
            writeln!(formatter, "{}", rule.join("-+-"))?;
        }
    }
    Ok(())
}

impl fmt::Display for InspectTable {
    /// The table: an `index` column, then one column per field, right-aligned; one line per element.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut header = vec!["index".to_owned()];
        header.extend(self.columns.iter().map(|column| column.name.clone()));
        let mut cells = vec![header];
        for (row, values) in self.rows.iter().enumerate() {
            let mut line = vec![(self.first + row).to_string()];
            line.extend(values.iter().map(Value::to_string));
            cells.push(line);
        }
        write_grid(formatter, &cells)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gust::StructFieldLayout;

    const U32: TypeLayout = TypeLayout::scalar("u32", ScalarKind::U32);
    static WORD: TypeLayout = TypeLayout::scalar("u32", ScalarKind::U32);

    static INNER_FIELDS: [StructFieldLayout; 2] = [
        StructFieldLayout {
            name: "p",
            offset: 0,
            ty: U32,
        },
        StructFieldLayout {
            name: "q",
            offset: 4,
            ty: TypeLayout::scalar("i32", ScalarKind::I32),
        },
    ];
    static INNER: TypeLayout = TypeLayout::structure("Inner", 8, 4, &INNER_FIELDS);
    static THREE: TypeLayout = TypeLayout::array(&WORD, 3);
    // `a` at 0, then a four-wide vector at 16 (the padding is words 1 to 3), a nested struct at 32, a 2x2 matrix at 40
    // (column stride 8) and `[u32; 3]` at 56: 68 bytes, stride 80.
    static FIELDS: [StructFieldLayout; 5] = [
        StructFieldLayout {
            name: "a",
            offset: 0,
            ty: TypeLayout::scalar("f32", ScalarKind::F32),
        },
        StructFieldLayout {
            name: "v",
            offset: 16,
            ty: TypeLayout::vector("vec4", ScalarKind::U32, 4),
        },
        StructFieldLayout {
            name: "inner",
            offset: 32,
            ty: INNER,
        },
        StructFieldLayout {
            name: "m",
            offset: 40,
            ty: TypeLayout::matrix("mat2", ScalarKind::U32, 2, 2),
        },
        StructFieldLayout {
            name: "arr",
            offset: 56,
            ty: THREE,
        },
    ];
    static LAYOUT: TypeLayout = TypeLayout::structure("Mixed", 68, 16, &FIELDS);

    /// Two elements; word `i` holds `100 + i`, except the `f32` at word 0 and the `i32` at word 9.
    fn planted() -> Vec<u8> {
        let mut words: Vec<u32> = (0..40).map(|i| 100 + i).collect();
        words[0] = 1.5_f32.to_bits();
        words[9] = (-9_i32) as u32;
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    #[test]
    fn nested_padded_matrix_and_array_fields_read_from_their_offsets() {
        let table = InspectTable::decode(&LAYOUT, 0, &planted(), 2).unwrap();
        assert_eq!(
            table.columns().collect::<Vec<_>>(),
            [
                "a", "v.x", "v.y", "v.z", "v.w", "inner.p", "inner.q", "m[0][0]", "m[0][1]",
                "m[1][0]", "m[1][1]", "arr[0]", "arr[1]", "arr[2]"
            ]
        );
        let row = |element: usize| -> Vec<String> {
            table.rows()[element].iter().map(Value::to_string).collect()
        };
        assert_eq!(
            row(0),
            [
                "1.5", "104", "105", "106", "107", "108", "-9", "110", "111", "112", "113", "114",
                "115", "116"
            ]
        );
        // The second element starts one stride (20 words) later: `v.x` is word 24, `m[1][0]` word 32, `arr[2]` 36.
        let second = row(1);
        assert_eq!(second[1], "124");
        assert_eq!(second[9], "132");
        assert_eq!(second[13], "136");
    }

    static INDEX_FIELDS: [StructFieldLayout; 1] = [StructFieldLayout {
        name: "index",
        offset: 0,
        ty: U32,
    }];
    static INDEXED: TypeLayout = TypeLayout::structure("Indexed", 4, 4, &INDEX_FIELDS);

    #[test]
    fn a_field_named_index_does_not_collide_with_the_element_index() {
        let bytes: Vec<u8> = [7_u32, 8].iter().flat_map(|v| v.to_le_bytes()).collect();
        let table = InspectTable::decode(&INDEXED, 10, &bytes, 2).unwrap();
        assert_eq!(
            table.to_json(),
            concat!(
                r#"{"type":"Indexed","first":10,"count":2,"fields":["index"],"rows":["#,
                r#"{"index":10,"values":{"index":7}},{"index":11,"values":{"index":8}}]}"#
            )
        );
    }

    #[test]
    fn a_scalar_buffer_has_the_one_column_value() {
        let table = InspectTable::decode(&U32, 0, &[5, 0, 0, 0], 1).unwrap();
        assert_eq!(table.columns().collect::<Vec<_>>(), ["value"]);
        assert_eq!(
            table.to_string(),
            "index | value\n------+------\n    0 |     5\n"
        );
    }

    #[test]
    fn json_escapes_and_spells_non_finite_floats_as_strings() {
        let mut out = String::new();
        json_string("a\"b\\c\n", &mut out);
        assert_eq!(out, r#""a\"b\\c\u000a""#);
        let mut number = String::new();
        Value::F32(f32::NEG_INFINITY).write_json(&mut number);
        assert_eq!(number, "\"-inf\"");
    }
}
