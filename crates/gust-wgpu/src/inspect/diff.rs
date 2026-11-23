//! The snapshot and its bit-for-bit diff.

use std::fmt;

use super::table::{InspectTable, Value};
use super::{InspectError, Inspector};
use crate::{GpuBuffer, HeadlessDevice};

/// One field of one element that differs between a snapshot and the buffer now.
#[derive(Clone, Debug)]
pub struct FieldChange {
    /// The column name.
    pub name: String,
    /// The value in the snapshot.
    pub before: Value,
    /// The value now.
    pub after: Value,
}

/// One element with at least one changed field.
#[derive(Clone, Debug)]
pub struct ElementChange {
    /// The element's index in the buffer.
    pub index: usize,
    /// The fields that differ.
    pub fields: Vec<FieldChange>,
}

/// The elements that differ between a [`Snapshot`] and the buffer now; bit-for-bit, so a NaN that stays the same NaN
/// is not a change.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Changes {
    /// The changed elements, by ascending index.
    pub elements: Vec<ElementChange>,
}

impl Changes {
    /// The indices of the changed elements.
    pub fn indices(&self) -> Vec<usize> {
        self.elements.iter().map(|element| element.index).collect()
    }
}

impl fmt::Display for Changes {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.elements.is_empty() {
            return writeln!(formatter, "no element changed");
        }
        for element in &self.elements {
            write!(formatter, "[{}]", element.index)?;
            for field in &element.fields {
                write!(
                    formatter,
                    " {}: {} -> {}",
                    field.name, field.before, field.after
                )?;
            }
            writeln!(formatter)?;
        }
        Ok(())
    }
}

/// A copy of a buffer's range taken by `Inspector::snapshot`, to be compared with the buffer later.
pub struct Snapshot<'a, T> {
    pub(super) buffer: &'a GpuBuffer<T>,
    pub(super) table: InspectTable,
}

impl<T> fmt::Debug for Snapshot<'_, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Snapshot")
            .field("table", &self.table)
            .finish_non_exhaustive()
    }
}

impl<T> Snapshot<'_, T> {
    /// The snapshot's rows.
    pub fn table(&self) -> &InspectTable {
        &self.table
    }

    /// Read the same range again and list the elements that differ. Blocks; not on `wasm32`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn changed(&self, device: &HeadlessDevice) -> Result<Changes, InspectError> {
        let table = Inspector::new(device, self.buffer)
            .range(self.range())
            .table()?;
        Ok(self.diff(&table))
    }

    /// `Snapshot::changed` without blocking.
    pub async fn changed_async(&self, device: &HeadlessDevice) -> Result<Changes, InspectError> {
        let table = Inspector::new(device, self.buffer)
            .range(self.range())
            .table_async()
            .await?;
        Ok(self.diff(&table))
    }

    fn range(&self) -> std::ops::Range<usize> {
        self.table.first..self.table.first + self.table.rows.len()
    }

    fn diff(&self, now: &InspectTable) -> Changes {
        diff_tables(&self.table, now)
    }
}

/// The elements of `before` that differ from the same rows of `now`, bit for bit.
fn diff_tables(before_table: &InspectTable, now: &InspectTable) -> Changes {
    let elements = before_table
        .rows
        .iter()
        .zip(&now.rows)
        .enumerate()
        .filter_map(|(row, (before, after))| {
            let fields: Vec<FieldChange> = before_table
                .columns
                .iter()
                .zip(before.iter().zip(after))
                .filter(|(_, (before, after))| !before.same_bits(**after))
                .map(|(column, (before, after))| FieldChange {
                    name: column.name.clone(),
                    before: *before,
                    after: *after,
                })
                .collect();
            (!fields.is_empty()).then_some(ElementChange {
                index: before_table.first + row,
                fields,
            })
        })
        .collect();
    Changes { elements }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gust::{ScalarKind, StructFieldLayout, TypeLayout};

    static FIELDS: [StructFieldLayout; 1] = [StructFieldLayout {
        name: "x",
        offset: 0,
        ty: TypeLayout::scalar("f32", ScalarKind::F32),
    }];
    static LAYOUT: TypeLayout = TypeLayout::structure("Cell", 4, 4, &FIELDS);

    fn table(values: &[f32]) -> InspectTable {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        InspectTable::decode(&LAYOUT, 0, &bytes, values.len()).unwrap()
    }

    #[test]
    fn the_diff_is_bit_for_bit() {
        // The same NaN twice is not a change; `0.0 -> -0.0` is, and so is a plain value change.
        let before = table(&[f32::NAN, 0.0, 1.0, 2.0]);
        let now = table(&[f32::NAN, -0.0, 1.0, 3.0]);
        let changes = diff_tables(&before, &now);
        assert_eq!(changes.indices(), [1, 3]);
        assert_eq!(
            changes.to_string(),
            "[1] x: 0.0 -> -0.0\n[3] x: 2.0 -> 3.0\n"
        );
        assert!(diff_tables(&before, &before).elements.is_empty());
    }
}
