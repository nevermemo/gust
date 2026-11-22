//! The line records of an emitted kernel and the table that builds them.

/// One inclusive, 1-based range of emitted Slang lines generated from one Rust
/// line (DX19). Records are in emission order, outer construct first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LineRecord {
    pub slang_start: u32,
    pub slang_end: u32,
    pub rust_line: u32,
}

/// Emitted Slang text and the line table recorded while writing it.
pub struct EmittedKernel {
    pub source: String,
    pub lines: Vec<LineRecord>,
}

/// Records which Rust line each emitted statement, helper and struct came from.
/// It only observes `output` and never changes what is written. Spans without a
/// location (line 0) are skipped, so a table can be empty.
#[derive(Default)]
pub(super) struct LineTable {
    pub(super) records: Vec<LineRecord>,
    /// Bytes of `output` already counted, and the newlines among them.
    scanned: usize,
    newlines: u32,
}

impl LineTable {
    /// The line the next write to `output` lands on.
    fn next_line(&mut self, output: &str) -> u32 {
        let fresh = &output.as_bytes()[self.scanned..];
        self.newlines += fresh.iter().filter(|byte| **byte == b'\n').count() as u32;
        self.scanned = output.len();
        self.newlines + 1
    }

    /// Open a record at the next line to be written.
    pub(super) fn begin(&mut self, output: &str, span: proc_macro2::Span) -> Option<usize> {
        let start = self.next_line(output);
        let rust_line = u32::try_from(span.start().line)
            .ok()
            .filter(|line| *line > 0)?;
        self.records.push(LineRecord {
            slang_start: start,
            slang_end: start,
            rust_line,
        });
        Some(self.records.len() - 1)
    }

    /// Close a record at the last line written so far.
    pub(super) fn finish_written(&mut self, output: &str, record: Option<usize>) {
        let end = self.next_line(output) - 1;
        self.set_end(record, end);
    }

    /// Close a record at the line about to be written (a closing brace).
    pub(super) fn finish_at_next(&mut self, output: &str, record: Option<usize>) {
        let end = self.next_line(output);
        self.set_end(record, end);
    }

    fn set_end(&mut self, record: Option<usize>, end: u32) {
        if let Some(record) = record.and_then(|index| self.records.get_mut(index)) {
            record.slang_end = end.max(record.slang_start);
        }
    }
}
