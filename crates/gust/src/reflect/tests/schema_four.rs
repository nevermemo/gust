use super::{fragment_support::*, *};

// ---- schema 4: arrays (slice m9.r2h) ----

/// A document of `schema` (3 or 4) for a compute entry point, whose parameters are the given JSON objects.
fn document_with_schema(schema: u32, parameters: &[String]) -> String {
    document3("compute", parameters, &[]).replacen(
        r#""schemaVersion":3"#,
        &format!(r#""schemaVersion":{schema}"#),
        1,
    )
}

const FLOAT_BODY: &str =
    r#"{"size":4,"alignment":4,"stride":4,"name":"float","kind":"scalar","scalarType":"float32"}"#;

/// `float[count]`, hand-written: size 4 * count, element stride 4.
fn float_array_body(count: u32) -> String {
    format!(
        r#"{{"size":{},"alignment":4,"stride":{},"name":"Array","kind":"array","elementCount":{count},"elementStride":4,"elementType":{FLOAT_BODY}}}"#,
        4 * count,
        4 * count
    )
}

/// `struct P { float a[4]; float2 b[2]; uint n; }` as a structured buffer, and `float[2][3]` as another; the numbers
/// are derived from the spike's per-type numbers (a at 0 size 16; b at 16 size 16, align 8, element stride 8; n at 32;
/// P size 40 align 8), not copied from a table row.
fn array_parameters() -> Vec<String> {
    let vec2 = r#"{"size":8,"alignment":8,"stride":8,"name":"vector","kind":"vector","scalarType":"float32","elementCount":2}"#;
    let b = format!(
        r#"{{"size":16,"alignment":8,"stride":16,"name":"Array","kind":"array","elementCount":2,"elementStride":8,"elementType":{vec2}}}"#
    );
    let uint = r#"{"size":4,"alignment":4,"stride":4,"name":"uint","kind":"scalar","scalarType":"uint32"}"#;
    let field = |name: &str, offset: u32, size: u32, ty: &str| {
        format!(
            r#"{{"name":"{name}","binding":{{"kind":"uniform","offset":{offset},"size":{size}}},"type":{ty}}}"#
        )
    };
    let p = format!(
        r#"{{"size":40,"alignment":8,"stride":40,"name":"P","kind":"struct","fields":[{},{},{}]}}"#,
        field("a", 0, 16, &float_array_body(4)),
        field("b", 16, 16, &b),
        field("n", 32, 4, uint)
    );
    let nested = format!(
        r#"{{"size":24,"alignment":4,"stride":24,"name":"Array","kind":"array","elementCount":2,"elementStride":12,"elementType":{}}}"#,
        float_array_body(3)
    );
    let buffer = |name: &str, index: u32, stride: u32, element: &str| {
        format!(
            r#"{{"name":"{name}","binding":{{"kind":"descriptorTableSlot","count":1,"index":{index},"space":0}},"elementStride":{stride},"type":{{"kind":"resource","baseShape":"structuredBuffer","access":"read","resultType":{element}}}}}"#
        )
    };
    vec![buffer("ps", 0, 40, &p), buffer("grid", 1, 24, &nested)]
}

fn array_of(ty: &ReflectedType) -> (&ReflectedType, u32, u32) {
    let ReflectedTypeKind::Array {
        element,
        count,
        element_stride,
    } = &ty.kind
    else {
        panic!("an array expected, got {ty:?}")
    };
    (element, *count, *element_stride)
}

#[test]
fn schema_four_reads_arrays_in_fields_nested_and_as_buffer_elements() {
    let compiled = read(
        &document_with_schema(4, &array_parameters()),
        &kernel_with(Vec::new()),
    )
    .unwrap();
    let reflection = &compiled.reflection;
    assert_eq!(reflection.schema_version, 4);
    assert_eq!(reflection.stage, Stage::Compute);
    let ReflectedTypeKind::Struct(fields) = &reflection.resources[0].element.kind else {
        panic!("a struct expected")
    };
    let a = &fields[0];
    assert_eq!((a.name.as_str(), a.offset, a.size), ("a", 0, 16));
    let (element, count, stride) = array_of(&a.ty);
    assert_eq!((count, stride), (4, 4));
    assert_eq!(element.kind, ReflectedTypeKind::Scalar(ScalarKind::F32));
    assert_eq!((a.ty.size, a.ty.alignment), (Some(16), Some(4)));
    let b = &fields[1];
    assert_eq!((b.offset, b.size, b.ty.alignment), (16, 16, Some(8)));
    let (element, count, stride) = array_of(&b.ty);
    assert_eq!((count, stride), (2, 8));
    assert_eq!(
        element.kind,
        ReflectedTypeKind::Vector {
            scalar: ScalarKind::F32,
            count: 2
        }
    );
    assert_eq!(fields[2].offset, 32);
    assert_eq!(reflection.resources[0].element_stride, Some(40));
    // A buffer element that is an array of arrays.
    let grid = &reflection.resources[1];
    assert_eq!(grid.element_stride, Some(24));
    let (inner, count, stride) = array_of(&grid.element);
    assert_eq!((count, stride), (2, 12));
    let (leaf, count, stride) = array_of(inner);
    assert_eq!((count, stride), (3, 4));
    assert_eq!(leaf.kind, ReflectedTypeKind::Scalar(ScalarKind::F32));
    assert_eq!((grid.element.size, inner.size), (Some(24), Some(12)));
}

#[test]
fn an_array_is_refused_below_schema_four_and_when_malformed() {
    let kernel = kernel_with(Vec::new());
    for schema in [2, 3] {
        let error = read(&document_with_schema(schema, &array_parameters()), &kernel)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("unsupported element type"),
            "{schema}: {error}"
        );
    }
    let one = || array_parameters().remove(1);
    for (bad, replacement, expected) in [
        (
            r#""elementCount":2,"elementStride":12"#,
            r#""elementCount":0,"elementStride":12"#,
            "positive elementCount",
        ),
        (
            r#""elementCount":2,"elementStride":12"#,
            r#""elementCount":2,"elementStride":0"#,
            "positive elementCount",
        ),
        (r#""elementCount":2,"#, "", "missing elementCount"),
        (r#""elementStride":12,"#, "", "missing elementStride"),
        (
            r#""elementType":"#,
            r#""elementTypo":"#,
            "missing elementType",
        ),
        // Lower bounds (hand-written): the outer element is 12 bytes of alignment 4, so a stride under 12 overlaps
        // the next element, 14 is off the alignment, and two elements 12 apart need 24 bytes, not 20.
        (
            r#""elementCount":2,"elementStride":12"#,
            r#""elementCount":2,"elementStride":8"#,
            "smaller than its element's size",
        ),
        (
            r#""elementCount":2,"elementStride":12"#,
            r#""elementCount":2,"elementStride":14"#,
            "not a multiple of its element's alignment",
        ),
        (
            r#""size":24,"#,
            r#""size":20,"#,
            "smaller than its elements need",
        ),
    ] {
        let parameter = one().replacen(bad, replacement, 1);
        assert_ne!(parameter, one(), "{bad}");
        let error = read(&document_with_schema(4, &[parameter]), &kernel).unwrap_err();
        assert!(
            matches!(error, Error::Invalid(_)) && error.to_string().contains(expected),
            "{bad}: {error}"
        );
    }
    // Only lower bounds are checked: the unpadded tight case (two elements 12 apart in exactly 24 bytes) must read.
    assert!(read(&document_with_schema(4, &[one()]), &kernel).is_ok());
    // Another unsupported kind is still refused under schema 4, with the schema-4 list of kinds.
    let parameter = one().replacen(r#""kind":"array""#, r#""kind":"texture""#, 1);
    let error = read(&document_with_schema(4, &[parameter]), &kernel)
        .unwrap_err()
        .to_string();
    assert!(error.contains("arrays and structs"), "{error}");
}

#[test]
fn schema_three_documents_are_read_as_before() {
    let compiled = read(
        &document_with_schema(3, &[storage_json("pts", 0, STORAGE_STRUCT, "read")]),
        &kernel_with(Vec::new()),
    )
    .unwrap();
    assert_eq!(compiled.reflection.schema_version, 3);
    let ReflectedTypeKind::Struct(fields) = &compiled.reflection.resources[0].element.kind else {
        panic!("a struct expected")
    };
    assert_eq!(fields[1].offset, 8);
}

#[test]
fn a_host_type_checked_against_an_array_is_a_mismatch_not_a_match() {
    // Until the array host layouts exist (slice r2) no descriptor names an array. A reflected `float[1]` has the
    // size 4, alignment 4 and stride 4 of `f32`, so every number coincides and only the type kind can refuse it.
    let parameter = format!(
        r#"{{"name":"one","binding":{{"kind":"descriptorTableSlot","count":1,"index":0,"space":0}},"elementStride":4,"type":{{"kind":"resource","baseShape":"structuredBuffer","access":"read","resultType":{}}}}}"#,
        float_array_body(1)
    );
    let compiled = read(
        &document_with_schema(4, &[parameter]),
        &kernel_with(Vec::new()),
    )
    .unwrap();
    let resource = &compiled.reflection.resources[0];
    let (element, count, stride) = array_of(&resource.element);
    assert_eq!((count, stride, element.size), (1, 4, Some(4)));
    assert_eq!(
        (
            resource.element.size,
            resource.element.alignment,
            resource.element.stride
        ),
        (Some(4), Some(4), Some(4))
    );
    let error = cross_check_pod::<f32>(resource).unwrap_err();
    assert!(
        matches!(&error, Error::Mismatch(message) if message.contains("type or field count differs")),
        "{error}"
    );
}

// m9.r2b: a host array layout against a schema-4 reflection (the documents are written from the host layout by
// `type_body`, so each mutation below is the one difference).

fn array_kernel(layout: TypeLayout) -> (String, KernelDescriptor) {
    let json = document_with_schema(4, &[storage_json("a", 0, layout, "read")]);
    let kernel = kernel_with(vec![parameter(
        "a",
        ParameterKind::Storage,
        Access::ReadOnly,
        ResourceLayout::storage_v2(layout),
        0,
    )]);
    (json, kernel)
}

#[test]
fn an_array_host_layout_matches_its_reflection_and_each_single_difference_is_refused() {
    // `Vec2 a[3]` as a buffer element: size 24, alignment 8, element stride 8 (spike P4b: float2[2] is 16/8/8).
    const THREE_V2: TypeLayout = TypeLayout::array(&VEC2, 3);
    let (json, kernel) = array_kernel(THREE_V2);
    assert_eq!(
        (THREE_V2.size, THREE_V2.alignment, THREE_V2.stride),
        (24, 8, 24)
    );
    let compiled = read(&json, &kernel).unwrap();
    let reports = compiled.reflection.cross_check_kernel(&kernel).unwrap();
    assert!(reports.iter().all(|report| report.is_full_abi_match()));
    let cases: &[(&str, &str, &str)] = &[
        // The reader only accepts a count that fits the size, so the count shrinks (a bigger one is refused when
        // the document is read).
        (
            r#""elementCount":3"#,
            r#""elementCount":2"#,
            "a.elementCount: host 3, compiler 2",
        ),
        (
            r#""scalarType":"float32","elementCount":2"#,
            r#""scalarType":"int32","elementCount":2"#,
            "a[]: type or field count differs",
        ),
        (
            r#""size":24,"alignment":8"#,
            r#""size":32,"alignment":8"#,
            "a.size: host 24, compiler 32",
        ),
        (
            r#""stride":8,"name":"Vec2""#,
            r#""stride":16,"name":"Vec2""#,
            "a[].stride: host 8, compiler 16",
        ),
    ];
    for (find, replace, expected) in cases {
        assert!(json.contains(find), "the fixture lost `{find}`");
        let edited = json.replacen(find, replace, 1);
        assert_eq!(&mismatch_of(&edited, &kernel), expected, "{find}");
    }
    // The element stride alone: with one element the reader accepts any stride from the element's size up, so the
    // size does not change with it.
    const ONE_V2: TypeLayout = TypeLayout::array(&VEC2, 1);
    let (json, kernel) = array_kernel(ONE_V2);
    let edited = json.replacen(
        r#""elementStride":8,"elementType""#,
        r#""elementStride":16,"elementType""#,
        1,
    );
    assert_ne!(edited, json);
    assert_eq!(
        mismatch_of(&edited, &kernel),
        "a.elementStride: host 8, compiler 16"
    );
    // A scalar array, and a matrix array (the element stride of a matrix is the whole matrix, 64).
    const FOUR_F: TypeLayout = TypeLayout::array(&F32_LAYOUT, 4);
    const TWO_M4: TypeLayout = TypeLayout::array(&MAT4, 2);
    assert_eq!(
        (FOUR_F.size, FOUR_F.alignment, TWO_M4.size, TWO_M4.alignment),
        (16, 4, 128, 16)
    );
    for layout in [FOUR_F, TWO_M4] {
        let (json, kernel) = array_kernel(layout);
        let compiled = read(&json, &kernel).unwrap();
        assert!(compiled.reflection.cross_check_kernel(&kernel).is_ok());
    }
    // The scalar array against a document that says the element is an unsigned integer.
    let (json, kernel) = array_kernel(FOUR_F);
    let edited = json.replacen(r#""scalarType":"float32""#, r#""scalarType":"uint32""#, 1);
    assert_eq!(
        mismatch_of(&edited, &kernel),
        "a[]: type or field count differs"
    );
}

#[test]
fn a_struct_with_array_fields_matches_its_reflection() {
    // struct S { float a[4]; float2 b[3]; }: a at 0 (16 bytes), b at 16 (24 bytes, alignment 8), size 40.
    const A: TypeLayout = TypeLayout::array(&F32_LAYOUT, 4);
    const B: TypeLayout = TypeLayout::array(&VEC2, 3);
    let s = make_struct("S", 40, 8, &[member("a", 0, A), member("b", 16, B)]);
    let (json, kernel) = array_kernel(s);
    let compiled = read(&json, &kernel).unwrap();
    let reports = compiled.reflection.cross_check_kernel(&kernel).unwrap();
    assert!(reports.iter().all(|report| report.is_full_abi_match()));
    assert_eq!(reports[0].checked_fields, 2);
    for (find, replace, expected) in [
        (
            r#""elementCount":4,"elementStride":4"#,
            r#""elementCount":3,"elementStride":4"#,
            "a.a.elementCount: host 4, compiler 3",
        ),
        (
            r#""scalarType":"float32","elementCount":2"#,
            r#""scalarType":"int32","elementCount":2"#,
            "a.b[]: type or field count differs",
        ),
        (
            r#""name":"b","binding":{"kind":"uniform","offset":16"#,
            r#""name":"b","binding":{"kind":"uniform","offset":24"#,
            "a.b.offset: host 16, compiler 24",
        ),
    ] {
        assert!(json.contains(find), "the fixture lost `{find}`");
        let edited = json.replacen(find, replace, 1);
        assert_eq!(&mismatch_of(&edited, &kernel), expected, "{find}");
    }
}

#[test]
fn an_array_host_layout_against_a_non_array_reflection_is_a_mismatch() {
    const FOUR_F: TypeLayout = TypeLayout::array(&F32_LAYOUT, 4);
    let (_, kernel) = array_kernel(FOUR_F);
    let json = document_with_schema(4, &[storage_json("a", 0, F32_LAYOUT, "read")]);
    let compiled = read(&json, &kernel).unwrap();
    assert!(compiled.reflection.cross_check_kernel(&kernel).is_err());
}
