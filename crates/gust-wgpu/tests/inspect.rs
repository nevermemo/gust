//! The buffer inspector (g2.insp): `HeadlessDevice::inspect` shows a struct buffer as a table with its field names,
//! the same range as JSON, per-field statistics with a planted NaN, and a snapshot diff after one dispatch. The
//! expected text is written out by hand from the values the test uploaded, not produced by the inspector.

mod common;

use std::ops::Bound;

use gust_wgpu::inspect::InspectError;
use gust_wgpu::{BufferBinding, Error, GpuBufferAccess};

#[gust::gpu]
mod world {
    pub struct Agent {
        pub id: uint,
        pub x: float,
        pub hp: int,
    }

    /// Every seventh element, from the third, takes a hit: `x` moves half a unit and `hp` drops by two.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn hit(id: SV_DispatchThreadID, mut agents: RWStructuredBuffer<Agent>) {
        let i = id.x;
        if i < agents.len() && i % 7 == 3 {
            let a = agents[i];
            agents[i] = Agent {
                id: a.id,
                x: a.x + 0.5,
                hp: a.hp - 2,
            };
        }
    }
}

const COUNT: usize = 20;

fn agents() -> Vec<world::Agent> {
    (0..COUNT)
        .map(|i| world::Agent {
            id: i as u32,
            x: i as f32 * 0.25,
            hp: 100 - i as i32,
        })
        .collect()
}

#[test]
fn the_table_has_field_names_and_a_range_from_the_middle() {
    let device = common::device_logged("inspect table");
    let buffer = device.create_typed_buffer("agents", &agents(), GpuBufferAccess::ReadWrite);
    let expected = "\
index | id |    x | hp
------+----+------+---
   10 | 10 |  2.5 | 90
   11 | 11 | 2.75 | 89
   12 | 12 |  3.0 | 88
";
    let table = device.inspect(&buffer).range(10..13).table().unwrap();
    assert_eq!(table.to_string(), expected);
    assert_eq!(table.columns().collect::<Vec<_>>(), ["id", "x", "hp"]);
    assert_eq!(table.first_index(), 10);
    assert_eq!(table.rows().len(), 3);
    // `println!("{}", device.inspect(&buffer).range(..))` prints the same table.
    assert_eq!(
        format!("{}", device.inspect(&buffer).range(10..13)),
        expected
    );
    // An open range is the whole buffer.
    assert_eq!(device.inspect(&buffer).table().unwrap().rows().len(), COUNT);
}

#[test]
fn the_json_carries_the_same_range() {
    let device = common::device_logged("inspect json");
    let buffer = device.create_typed_buffer("agents", &agents(), GpuBufferAccess::ReadWrite);
    let json = device.inspect(&buffer).range(10..13).to_json().unwrap();
    assert_eq!(
        json,
        concat!(
            r#"{"type":"Agent","first":10,"count":3,"fields":["id","x","hp"],"rows":["#,
            r#"{"index":10,"values":{"id":10,"x":2.5,"hp":90}},"#,
            r#"{"index":11,"values":{"id":11,"x":2.75,"hp":89}},"#,
            r#"{"index":12,"values":{"id":12,"x":3.0,"hp":88}}]}"#
        )
    );
}

#[test]
fn stats_count_the_planted_nan_and_leave_it_out_of_min_and_max() {
    let device = common::device_logged("inspect stats");
    let mut values = agents();
    values[5].x = f32::NAN;
    values[9].x = -4.5;
    values[12].x = 100.0;
    let buffer = device.create_typed_buffer("agents", &values, GpuBufferAccess::ReadWrite);
    let stats = device.inspect(&buffer).stats().unwrap();
    assert_eq!((stats.first, stats.elements), (0, COUNT));

    let x = stats.field("x").unwrap();
    assert_eq!(x.nan_count, Some(1));
    assert_eq!(x.count, COUNT - 1);
    assert_eq!(x.min.unwrap().to_string(), "-4.5");
    assert_eq!(x.max.unwrap().to_string(), "100.0");

    let id = stats.field("id").unwrap();
    assert_eq!(id.nan_count, None);
    assert_eq!(id.count, COUNT);
    assert_eq!(id.min.unwrap().to_string(), "0");
    assert_eq!(id.max.unwrap().to_string(), "19");
    let hp = stats.field("hp").unwrap();
    assert_eq!(
        (hp.min.unwrap().to_string(), hp.max.unwrap().to_string()),
        ("81".to_owned(), "100".to_owned())
    );

    // A range counts only its own elements: the NaN at 5 is outside 6..9.
    let inner = device.inspect(&buffer).range(6..9).stats().unwrap();
    assert_eq!(inner.field("x").unwrap().nan_count, Some(0));
    assert_eq!(inner.elements, 3);
    // The text form lists the NaN count of the float column and a dash for the integer ones.
    assert_eq!(
        stats.to_string(),
        "\
field | count |  min |   max | nan
------+-------+------+-------+----
   id |    20 |    0 |    19 |   -
    x |    19 | -4.5 | 100.0 |   1
   hp |    20 |   81 |   100 |   -
"
    );
}

#[test]
fn a_snapshot_diff_finds_exactly_the_elements_one_dispatch_changed() {
    let device = common::device_logged("inspect snapshot");
    let buffer = device.create_typed_buffer("agents", &agents(), GpuBufferAccess::ReadWrite);
    let whole = device.inspect(&buffer).snapshot().unwrap();
    // The inner range 4..10 holds no element the kernel changes (3, 10 and 17 do).
    let inner = device.inspect(&buffer).range(4..10).snapshot().unwrap();
    assert!(whole.changed(&device).unwrap().elements.is_empty());

    device
        .dispatch_buffers(
            &world::hit::DESCRIPTOR,
            COUNT as u32,
            &[BufferBinding::read_write(&buffer)],
        )
        .unwrap();

    let changes = whole.changed(&device).unwrap();
    assert_eq!(changes.indices(), [3, 10, 17]);
    let first = &changes.elements[0];
    let fields: Vec<_> = first
        .fields
        .iter()
        .map(|field| {
            (
                field.name.as_str(),
                field.before.to_string(),
                field.after.to_string(),
            )
        })
        .collect();
    assert_eq!(
        fields,
        [
            ("x", "0.75".to_owned(), "1.25".to_owned()),
            ("hp", "97".to_owned(), "95".to_owned()),
        ]
    );
    assert_eq!(
        changes.to_string(),
        "[3] x: 0.75 -> 1.25 hp: 97 -> 95\n\
         [10] x: 2.5 -> 3.0 hp: 90 -> 88\n\
         [17] x: 4.25 -> 4.75 hp: 83 -> 81\n"
    );
    assert!(inner.changed(&device).unwrap().elements.is_empty());
    assert_eq!(
        inner.changed(&device).unwrap().to_string(),
        "no element changed\n"
    );
}

#[test]
fn a_buffer_that_is_not_readable_returns_the_readback_error_with_a_hint() {
    let device = common::device_logged("inspect refusal");
    let buffer = device.create_typed_buffer("agents", &agents(), GpuBufferAccess::ReadOnly);
    for error in [
        device.inspect(&buffer).table().unwrap_err(),
        device.inspect(&buffer).stats().unwrap_err(),
        device.inspect(&buffer).snapshot().unwrap_err(),
        device.inspect(&buffer).to_json().unwrap_err(),
    ] {
        assert!(
            matches!(
                error,
                InspectError::Readback(Error::PersistentBufferNotReadable)
            ),
            "{error:?}"
        );
        assert_eq!(
            error.to_string(),
            "readback requires a persistent buffer created with read-write access; \
             hint: create the buffer with `GpuBufferAccess::ReadWrite` to inspect it"
        );
    }
    // Printing cannot return the error, so it prints it.
    assert!(
        format!("{}", device.inspect(&buffer)).starts_with("<inspect failed: readback requires")
    );
}

#[test]
fn a_range_outside_the_buffer_is_refused_and_an_empty_one_is_an_empty_table() {
    let device = common::device_logged("inspect range");
    let buffer = device.create_typed_buffer("agents", &agents(), GpuBufferAccess::ReadWrite);
    let error = device.inspect(&buffer).range(18..21).table().unwrap_err();
    assert!(
        matches!(
            error,
            InspectError::Range {
                start: 18,
                end: 21,
                len: COUNT
            }
        ),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        "inspect range 18..21 does not fit a buffer of 20 elements"
    );
    let (from, to) = (5, 3);
    let backwards = device.inspect(&buffer).range(from..to).table().unwrap_err();
    assert!(
        matches!(
            backwards,
            InspectError::Range {
                start: 5,
                end: 3,
                len: COUNT
            }
        ),
        "{backwards:?}"
    );
    // An excluded start bound skips that element: (1, 3] is elements 2 and 3.
    let open = device
        .inspect(&buffer)
        .range((Bound::Excluded(1), Bound::Included(3)))
        .table()
        .unwrap();
    assert_eq!((open.first_index(), open.rows().len()), (2, 2));
    let empty = device.inspect(&buffer).range(4..4).table().unwrap();
    assert!(empty.rows().is_empty());
    assert_eq!(
        empty.to_string(),
        "index | id | x | hp\n------+----+---+---\n"
    );
}

#[test]
fn a_scalar_buffer_has_the_one_column_value_and_the_async_form_agrees() {
    let device = common::device_logged("inspect async");
    let buffer = device.create_typed_buffer("values", &[7u32, 8, 9], GpuBufferAccess::ReadWrite);
    let table = device.inspect(&buffer).range(1..).table().unwrap();
    assert_eq!(table.columns().collect::<Vec<_>>(), ["value"]);
    assert_eq!(
        table.to_string(),
        "index | value\n------+------\n    1 |     8\n    2 |     9\n"
    );
    let async_json = common::drive(device.inspect(&buffer).range(1..).to_json_async()).unwrap();
    assert_eq!(
        async_json,
        r#"{"type":"u32","first":1,"count":2,"fields":["value"],"rows":[{"index":1,"values":{"value":8}},{"index":2,"values":{"value":9}}]}"#
    );
}
