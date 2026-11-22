//! `#[pad]` (m14.pad, D233 and D239): the fields inserted for each layout case, the generated `new`, the Slang the
//! padded struct lowers to, the expansion's layout assertions, GUST-E0219 and the recovery that keeps a padded module
//! consistent beside another error.

use quote::ToTokens;

use crate::{
    diagnostics::{
        Code,
        testing::{assert_code, assert_rendered_code, parse_str},
    },
    expand::{self, kernel_options, pad},
    profile::GRID,
    slang::emit_kernel,
    validate::validate_module,
};

/// A module with `items` and a kernel that reads the uniform `u: Uniform<S>`.
fn uniform_module(items: &str) -> String {
    format!(
        "mod m {{ {items} #[kernel] pub fn run(id: SV_DispatchThreadID, u: Uniform<S>, \
         mut out: RWStructuredBuffer<float>) {{ let i = id.x; out[i] = 1.0; }} }}"
    )
}

/// `source` after the pad pass, and the constructors it made.
fn padded(source: &str) -> syn::Result<(syn::ItemMod, String)> {
    let module: syn::ItemMod = parse_str(source)?;
    let (module, constructors) = pad::apply(module, &GRID)?;
    Ok((module, constructors.to_string()))
}

/// The fields of the struct `name` of `module` as `name: Type` texts.
fn fields_of(module: &syn::ItemMod, name: &str) -> Vec<String> {
    let items = &module.content.as_ref().unwrap().1;
    let item = items
        .iter()
        .find_map(|item| match item {
            syn::Item::Struct(item) if item.ident == name => Some(item),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no struct {name}"));
    assert!(
        !item.attrs.iter().any(|a| a.path().is_ident("pad")),
        "`#[pad]` is left on `{name}`"
    );
    item.fields
        .iter()
        .map(|field| {
            format!(
                "{}: {}",
                field.ident.as_ref().unwrap(),
                field.ty.to_token_stream()
            )
        })
        .collect()
}

#[test]
fn the_inserted_fields_for_each_layout_case() {
    let four = "x: f32, y: f32, z: f32, w: f32";
    // (items, the struct S as written, the fields S has after the pass)
    let cases: Vec<(String, &str, Vec<&str>)> = vec![
        // A tail: the size rounds up to 16.
        (
            String::new(),
            "a: f32",
            vec!["a: f32", "_pad0: f32", "_pad1: f32", "_pad2: f32"],
        ),
        (
            String::new(),
            "a: Vec2",
            vec!["a: Vec2", "_pad0: f32", "_pad1: f32"],
        ),
        (
            String::new(),
            "a: f32, b: f32, c: f32",
            vec!["a: f32", "b: f32", "c: f32", "_pad0: f32"],
        ),
        // A gap the field's own alignment causes (GUST-E0208 without the attribute).
        (
            String::new(),
            "a: f32, v: Vec2",
            vec!["a: f32", "_pad0: f32", "v: Vec2"],
        ),
        // A gap and a tail in one struct.
        (
            String::new(),
            "a: f32, v: Vec2, b: f32",
            vec![
                "a: f32",
                "_pad0: f32",
                "v: Vec2",
                "b: f32",
                "_pad1: f32",
                "_pad2: f32",
                "_pad3: f32",
            ],
        ),
        // Nothing to insert: whole rows.
        (String::new(), "v: Vec4", vec!["v: Vec4"]),
        (
            String::new(),
            "a: Vec2, b: Vec2",
            vec!["a: Vec2", "b: Vec2"],
        ),
        // A gap before a nested struct (16-byte start).
        (
            format!("struct Inner {{ {four} }}"),
            "a: f32, inner: Inner",
            vec![
                "a: f32",
                "_pad0: f32",
                "_pad1: f32",
                "_pad2: f32",
                "inner: Inner",
            ],
        ),
        // A tail after a nested struct, which is not padded itself (it keeps its own GUST-E0209).
        (
            "struct Inner { x: f32, y: f32, z: f32, w: f32 }".to_owned(),
            "inner: Inner, a: f32",
            vec![
                "inner: Inner",
                "a: f32",
                "_pad0: f32",
                "_pad1: f32",
                "_pad2: f32",
            ],
        ),
        // A matrix after a scalar: `m` starts at 16 (a 12-byte gap), 64 bytes long, no tail.
        (
            String::new(),
            "a: f32, m: Mat4",
            vec![
                "a: f32",
                "_pad0: f32",
                "_pad1: f32",
                "_pad2: f32",
                "m: Mat4",
            ],
        ),
        // A name the user has: the first free numbers.
        (
            String::new(),
            "_pad0: f32",
            vec!["_pad0: f32", "_pad1: f32", "_pad2: f32", "_pad3: f32"],
        ),
        (
            String::new(),
            "a: f32, _pad1: f32",
            vec!["a: f32", "_pad1: f32", "_pad0: f32", "_pad2: f32"],
        ),
    ];
    for (items, fields, expected) in cases {
        let source = uniform_module(&format!("{items} #[pad] struct S {{ {fields} }}"));
        let (module, _) = padded(&source).unwrap_or_else(|error| panic!("{fields}: {error}"));
        assert_eq!(fields_of(&module, "S"), expected, "{fields}");
        // The padded struct is padding-free in the constant-buffer layout, as the validator reads it.
        if items.is_empty() {
            validate_module(&module).unwrap_or_else(|error| panic!("{fields}: {error}"));
        }
    }
}

#[test]
fn a_nested_struct_that_is_marked_too_is_padded_first_so_the_outer_sees_its_size() {
    let source = uniform_module(
        "#[pad] struct Inner { x: f32, y: f32, z: f32 } #[pad] struct S { a: f32, inner: Inner }",
    );
    let (module, constructors) = padded(&source).unwrap();
    assert_eq!(
        fields_of(&module, "Inner"),
        ["x: f32", "y: f32", "z: f32", "_pad0: f32"]
    );
    // `inner` starts at 16 after `a`: 12 bytes of padding; it ends at 32, a multiple of 16.
    assert_eq!(
        fields_of(&module, "S"),
        [
            "a: f32",
            "_pad0: f32",
            "_pad1: f32",
            "_pad2: f32",
            "inner: Inner"
        ]
    );
    validate_module(&module).unwrap();
    // Declaration order of the structs does not matter: the outer one may come first.
    let reversed = uniform_module(
        "#[pad] struct S { a: f32, inner: Inner } #[pad] struct Inner { x: f32, y: f32, z: f32 }",
    );
    let (module, _) = padded(&reversed).unwrap();
    assert_eq!(fields_of(&module, "S").len(), 5);
    assert_eq!(fields_of(&module, "Inner").len(), 4);
    // One constructor per marked struct, in dependency order.
    assert_eq!(
        constructors.matches("pub const fn new").count(),
        2,
        "{constructors}"
    );
    assert!(
        constructors.find("impl Inner").unwrap() < constructors.find("impl S").unwrap(),
        "{constructors}"
    );
}

#[test]
fn the_generated_new_takes_the_user_fields_in_declaration_order_and_zeroes_the_padding() {
    let source = uniform_module("#[pad] struct S { a: f32, v: Vec2, b: f32 }");
    let (_, constructors) = padded(&source).unwrap();
    assert_eq!(
        constructors,
        "impl S { # [doc = \"Builds a `S` from its fields, in declaration order; the padding GUST inserted \
         (`#[pad]`) is zero.\"] # [allow (clippy :: too_many_arguments)] pub const fn new (a : f32 , v : Vec2 , b : f32) -> Self \
         { Self { a , v , b , _pad0 : 0.0 , _pad1 : 0.0 , _pad2 : 0.0 , _pad3 : 0.0 , } } }"
    );
    // A struct with nothing to pad still gets one, and one with a user-named `_pad0` keeps it as a parameter.
    let (_, whole) = padded(&uniform_module("#[pad] struct S { v: Vec4 }")).unwrap();
    assert!(
        whole.contains("pub const fn new (v : Vec4) -> Self { Self { v , } }"),
        "{whole}"
    );
    let (_, named) = padded(&uniform_module("#[pad] struct S { _pad0: f32 }")).unwrap();
    assert!(
        named.contains(
            "new (_pad0 : f32) -> Self { Self { _pad0 , _pad1 : 0.0 , _pad2 : 0.0 , _pad3 : 0.0 , } }"
        ),
        "{named}"
    );
}

#[test]
fn a_module_without_the_attribute_is_untouched() {
    let source = uniform_module("struct S { a: f32, b: f32, c: f32, d: f32 }");
    let module: syn::ItemMod = parse_str(&source).unwrap();
    let before = module.to_token_stream().to_string();
    let (after, constructors) = pad::apply(module, &GRID).unwrap();
    assert_eq!(after.to_token_stream().to_string(), before);
    assert!(constructors.is_empty());
}

/// The module `source` expanded as `#[gpu]` does, and the tokens as text.
fn expanded(source: &str) -> syn::Result<String> {
    let module: syn::ItemMod = parse_str(source)?;
    expand::expand(module).map(|tokens| tokens.to_string())
}

#[test]
fn the_padded_struct_reaches_the_layout_assertions_gpupod_and_the_constructor() {
    let text = expanded(&uniform_module("#[pad] struct S { a: f32, v: Vec2 }")).unwrap();
    // The struct: private pad field, no attribute left, `repr(C, align(16))` (it is uniform-reached).
    assert!(!text.contains("# [pad]"), "{text}");
    assert!(text.contains("_pad0 : f32"), "{text}");
    assert!(text.contains("repr (C , align (16))"), "{text}");
    // The `GpuPod` impl lists the pad field with its real offset, and the layout assertions name it.
    assert!(text.contains("offset_of ! (S , _pad0)"), "{text}");
    assert!(text.contains("name : \"_pad0\""), "{text}");
    assert!(
        text.contains("field `_pad0` of `S` is not at byte 4 of the constant-buffer layout"),
        "{text}"
    );
    assert!(
        text.contains("field `v` of `S` is not at byte 8 of the constant-buffer layout"),
        "{text}"
    );
    assert!(
        text.contains("`S` is not 16 bytes, the size of its constant-buffer layout"),
        "{text}"
    );
    // The constructor sits inside the generated module and is Rust only: no Slang text names it.
    assert!(
        text.contains("pub const fn new (a : f32 , v : Vec2) -> Self"),
        "{text}"
    );
}

/// The Slang of the kernel `run` of the padded `source`.
fn slang_of(source: &str) -> String {
    let (module, _) = padded(source).unwrap();
    validate_module(&module).unwrap();
    let kernel = module
        .content
        .as_ref()
        .unwrap()
        .1
        .iter()
        .find_map(|item| match item {
            syn::Item::Fn(function) if function.sig.ident == "run" => Some(function),
            _ => None,
        })
        .unwrap();
    let options = kernel_options(kernel).unwrap().unwrap();
    emit_kernel(&module, kernel, options)
        .unwrap()
        .source
        .replace("\r\n", "\n")
}

#[test]
fn the_pad_fields_reach_slang_unchanged() {
    let slang = slang_of(&uniform_module(
        "struct Inner { x: f32, y: f32, z: f32, w: f32 } #[pad] struct S { a: f32, inner: Inner, b: f32 }",
    ));
    let block = slang
        .split("struct S")
        .nth(1)
        .and_then(|rest| rest.split('}').next())
        .unwrap_or_else(|| panic!("no struct S: {slang}"));
    let fields: Vec<&str> = block
        .lines()
        .map(str::trim)
        .filter(|l| l.ends_with(';'))
        .collect();
    assert_eq!(
        fields,
        [
            "float a;",
            "float _pad0;",
            "float _pad1;",
            "float _pad2;",
            "Inner inner;",
            "float b;",
            "float _pad3;",
            "float _pad4;",
            "float _pad5;",
        ],
        "{slang}"
    );
}

#[test]
fn misuse_of_the_attribute_is_e0219() {
    let kernel = "#[kernel] pub fn run(id: SV_DispatchThreadID, u: Uniform<S>, \
                  mut out: RWStructuredBuffer<float>) { out[id.x] = 1.0; }";
    let four = "a: f32, b: f32, c: f32, d: f32";
    // (source, a word the message must hold)
    let cases = [
        // Arguments.
        (uniform_module("#[pad(4)] struct S { a: f32 }"), "takes no arguments"),
        (uniform_module("#[pad = \"x\"] struct S { a: f32 }"), "takes no arguments"),
        // Not a struct.
        (format!("mod m {{ struct S {{ {four} }} #[pad] fn f() {{ }} {kernel} }}"), "goes on a struct"),
        (format!("mod m {{ struct S {{ {four} }} #[pad] enum E {{ A }} {kernel} }}"), "goes on a struct"),
        (format!("mod m {{ struct S {{ {four} }} #[pad] const N: u32 = 1; {kernel} }}"), "goes on a struct"),
        (format!("mod m {{ struct S {{ {four} }} #[pad] impl S {{ }} {kernel} }}"), "goes on a struct"),
        (format!("mod m {{ struct S {{ {four} }} #[pad] #[kernel] pub fn other(id: SV_DispatchThreadID) {{ }} {kernel} }}"), "goes on a struct"),
        // A struct without named fields.
        (uniform_module("#[pad] struct S(f32);"), "named fields"),
        // A struct no uniform reaches.
        (
            format!("mod m {{ struct S {{ {four} }} #[pad] struct T {{ a: f32 }} {kernel} }}"),
            "no uniform parameter",
        ),
        (
            "mod m { #[pad] struct P { a: f32 } #[kernel] pub fn run(id: SV_DispatchThreadID, \
             b: StructuredBuffer<P>, mut out: RWStructuredBuffer<float>) { out[id.x] = b[id.x].a; } }"
                .to_owned(),
            "only a storage buffer reaches it",
        ),
        (
            "mod m { #[pad] struct P { a: f32 } #[kernel] pub fn run(id: SV_DispatchThreadID, \
             mut out: RWStructuredBuffer<float>) { out[id.x] = 1.0; } }"
                .to_owned(),
            "no uniform parameter",
        ),
        // A `new` of the user's in the module.
        (
            uniform_module(&format!(
                "#[pad] struct S {{ {four} }} impl S {{ fn new() -> S {{ S {{ a: 0.0, b: 0.0, c: 0.0, d: 0.0 }} }} }}"
            )),
            "would clash",
        ),
    ];
    for (source, word) in cases {
        let error = expanded(&source).expect_err(&source);
        assert_code(&error, Code::PadMisuse);
        assert!(error.to_string().contains(word), "{word}: {error}");
    }
    // A `new` of a trait impl, or of another struct, is no clash.
    expanded(&uniform_module(&format!(
        "#[pad] struct S {{ {four} }} struct Other {{ a: f32 }} impl Other {{ fn new() -> f32 {{ 0.0 }} }}"
    )))
    .unwrap();
    // A uniform struct reached through a field is reached too (it need not be the parameter's own element).
    expanded(&uniform_module(
        "#[pad] struct Inner { x: f32 } struct S { inner: Inner, v: Vec4, w: Vec4 }",
    ))
    .unwrap();
}

#[test]
fn a_struct_without_the_attribute_is_still_e0209_and_the_message_names_it() {
    for fields in [
        "a: f32",
        "a: f32, b: f32, c: f32",
        "a: Vec2, b: Vec2, c: Vec2",
    ] {
        let error = expanded(&uniform_module(&format!("struct S {{ {fields} }}"))).unwrap_err();
        assert_code(&error, Code::UniformPadding);
        assert!(error.to_string().contains("`#[pad]`"), "{error}");
    }
    let gap = expanded(&uniform_module(
        "struct Inner { x: f32, y: f32, z: f32, w: f32 } struct S { a: f32, inner: Inner }",
    ))
    .unwrap_err();
    assert_code(&gap, Code::UniformPadding);
    assert!(gap.to_string().contains("`#[pad]`"), "{gap}");
    // The nested struct that the marked one holds is not padded by the mark: its own gap is E0209.
    let inner = expanded(&uniform_module(
        "struct Inner { x: f32, y: f32, z: f32 } #[pad] struct S { a: f32, inner: Inner }",
    ))
    .unwrap_err();
    assert_code(&inner, Code::UniformPadding);
}

#[test]
fn e0208_for_a_struct_a_uniform_reaches_names_the_attribute_and_for_a_storage_struct_does_not() {
    // Rust's own alignment leaves the gap and the tail, so E0208 comes before E0209.
    for fields in ["a: f32, v: Vec2", "v: Vec2, a: f32"] {
        let error = expanded(&uniform_module(&format!("struct S {{ {fields} }}"))).unwrap_err();
        assert_code(&error, Code::ImplicitPadding);
        assert!(
            error
                .to_string()
                .contains("mark the struct `#[pad]` and GUST inserts the padding"),
            "{fields}: {error}"
        );
    }
    let storage = expanded(
        "mod m { struct S { a: f32, v: Vec2 } #[kernel] pub fn run(id: SV_DispatchThreadID, \
         b: StructuredBuffer<S>, mut out: RWStructuredBuffer<float>) { out[id.x] = b[id.x].a; } }",
    )
    .unwrap_err();
    assert_code(&storage, Code::ImplicitPadding);
    assert!(!storage.to_string().contains("#[pad]"), "{storage}");
}

#[test]
fn fields_the_layout_cannot_hold_keep_their_own_errors_under_the_attribute() {
    // `Mat2` in a marked struct is E0205 and an array is E0215: the pass leaves such a struct alone.
    let mat = expanded(&uniform_module("#[pad] struct S { m: Mat2, v: Vec2 }")).unwrap_err();
    assert_code(&mat, Code::UniformFieldType);
    let array = expanded(&uniform_module("#[pad] struct S { w: [f32; 4] }")).unwrap_err();
    assert_code(&array, Code::ArrayInUniform);
}

/// The output of `#[gpu]` on `source`, as text.
fn gpu_text(source: &str) -> String {
    let item: proc_macro2::TokenStream = source.parse().unwrap();
    crate::gpu_impl_with(
        proc_macro2::TokenStream::new(),
        item,
        expand::Profile::Other,
    )
    .to_string()
}

#[test]
fn a_module_with_an_error_elsewhere_still_compiles_its_padded_structs_and_new() {
    // The error is a `f64` local of the kernel; the struct is padded and has its `new`, and no `#[pad]` is left.
    let source = "mod m { #[pad] struct S { a: f32 } #[kernel] pub fn run(id: SV_DispatchThreadID, \
                  u: Uniform<S>, mut out: RWStructuredBuffer<float>) { let x: f64 = 1.0; out[id.x] = u.a; } }";
    let text = gpu_text(source);
    assert!(text.contains("compile_error"), "{text}");
    assert!(!text.contains("# [pad]"), "{text}");
    assert!(text.contains("_pad2 : f32"), "{text}");
    assert!(
        text.contains("pub const fn new (a : f32) -> Self"),
        "{text}"
    );
    // A misuse of the attribute is the one error, and the attribute is dropped from the module handed back.
    let misused = gpu_text(
        "mod m { #[pad] struct P { a: f32 } #[kernel] pub fn run(id: SV_DispatchThreadID, \
         mut out: RWStructuredBuffer<float>) { out[id.x] = 1.0; } }",
    );
    assert_rendered_code(&misused, Code::PadMisuse);
    assert!(!misused.contains("# [pad]"), "{misused}");
    assert!(!misused.contains("_pad0"), "{misused}");
    // A struct reached by the second kernel only. (A `#[variant]` has its kernel's signature, token for token, so a
    // variant cannot reach a struct its kernel does not; `variant::lower` runs before the pad pass, so both are
    // kernels there. A variant-only reach case therefore cannot exist and this is a plain second kernel.)
    let variant = gpu_text(
        "mod m { #[pad] struct S { a: f32 } \
         #[kernel] pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<float>) { out[id.x] = 1.0; } \
         #[kernel] pub fn other(id: SV_DispatchThreadID, u: Uniform<S>, mut out: RWStructuredBuffer<float>) \
         { let x: f64 = 1.0; out[id.x] = u.a; } }",
    );
    assert!(variant.contains("_pad0 : f32"), "{variant}");
}
