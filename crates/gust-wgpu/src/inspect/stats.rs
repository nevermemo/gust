//! Per-field statistics.

use std::fmt;

use super::table::{InspectTable, Kind, Value, write_grid};

/// Statistics of one column over the inspected range.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct FieldStats {
    /// The column name.
    pub name: String,
    /// How many values were compared: every element, less the NaNs of a float column.
    pub count: usize,
    /// The smallest value, or `None` when `count` is zero.
    pub min: Option<Value>,
    /// The largest value, or `None` when `count` is zero.
    pub max: Option<Value>,
    /// The number of NaNs for a float column; `None` for an integer column.
    pub nan_count: Option<usize>,
}

/// Per-field statistics: minimum, maximum and, for floats, the NaN count.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct InspectStats {
    /// The buffer index of the first element counted.
    pub first: usize,
    /// How many elements were counted.
    pub elements: usize,
    /// One entry per column, in layout order.
    pub fields: Vec<FieldStats>,
}

impl InspectStats {
    pub(super) fn of(table: &InspectTable) -> Self {
        let fields = table
            .columns
            .iter()
            .enumerate()
            .map(|(position, column)| {
                let mut stats = FieldStats {
                    name: column.name.clone(),
                    count: 0,
                    min: None,
                    max: None,
                    nan_count: matches!(column.kind, Kind::F32).then_some(0),
                };
                for row in &table.rows {
                    let value = row[position];
                    if value.is_nan() {
                        if let Some(nan) = stats.nan_count.as_mut() {
                            *nan += 1;
                        }
                        continue;
                    }
                    stats.count += 1;
                    if stats.min.is_none_or(|min| value.total_cmp(min).is_lt()) {
                        stats.min = Some(value);
                    }
                    if stats.max.is_none_or(|max| value.total_cmp(max).is_gt()) {
                        stats.max = Some(value);
                    }
                }
                stats
            })
            .collect();
        Self {
            first: table.first,
            elements: table.rows.len(),
            fields,
        }
    }

    /// The entry of the column called `name`.
    pub fn field(&self, name: &str) -> Option<&FieldStats> {
        self.fields.iter().find(|field| field.name == name)
    }
}

impl fmt::Display for InspectStats {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let show = |value: Option<Value>| value.map_or("-".to_owned(), |value| value.to_string());
        let mut cells = vec![
            ["field", "count", "min", "max", "nan"]
                .map(str::to_owned)
                .to_vec(),
        ];
        for field in &self.fields {
            cells.push(vec![
                field.name.clone(),
                field.count.to_string(),
                show(field.min),
                show(field.max),
                field
                    .nan_count
                    .map_or("-".to_owned(), |nan| nan.to_string()),
            ]);
        }
        write_grid(formatter, &cells)
    }
}
