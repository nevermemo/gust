//! The generated `capture` of a kernel (g2.cap): its signature, the parameters it copies and how it rebinds them for the
//! replay, the cases that return an error (no twin, a texture) and the profile that has none.

use crate::expand::{Environment, Profile, expand_in};

/// A kernel with a struct element, a uniform, a read-only and a read-write buffer, `id` not first.
const PARTICLES: &str = "mod sim { \
    pub struct Particle { pub pos: float, pub vel: float, pub tag: uint } \
    pub struct Params { pub dt: float, pub gain: float, pub pad0: float, pub pad1: float } \
    #[kernel] \
    pub fn step(params: Uniform<Params>, input: StructuredBuffer<Particle>, \
                mut out: RWStructuredBuffer<Particle>, id: SV_DispatchThreadID) { \
        let i = id.x; \
        if i < out.len() { let p = input[i]; out[i] = Particle { pos: p.pos + p.vel * params.dt, vel: p.vel * params.gain, tag: p.tag + 1u32 }; } \
    } \
}";

fn expansion_text(source: &str, profile: Profile) -> String {
    let module: syn::ItemMod = crate::diagnostics::testing::parse_str(source).unwrap();
    crate::expand::expand_with(module, profile)
        .unwrap()
        .to_string()
}

/// The text of the gated `capture` item of `kernel`.
fn capture_item(text: &str, kernel: &str) -> String {
    let marker = format!("# [cfg (any (test , gust_dual))] impl {kernel} {{");
    let start = text.find(&marker).expect("a capture item");
    let rest = &text[start..];
    let mut depth = 0_usize;
    for (offset, character) in rest.char_indices() {
        match character {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return rest[..=offset].to_owned();
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced capture item in {text}");
}

#[test]
fn a_capture_is_generated_once_beside_the_twin_under_the_same_gate() {
    let text = expansion_text(PARTICLES, Profile::Other);
    assert_eq!(text.matches("pub fn capture <").count(), 1, "{text}");
    let item = capture_item(&text, "step");
    // The signature: any device that can copy buffers, and the call that `call` builds.
    assert!(
        item.contains(
            "pub fn capture < 'r , __GustD > (__gust_device : & __GustD , __gust_call : & :: gust :: Call < 'r , __GustD , Self , 3 > ,)"
        ),
        "{item}"
    );
    assert!(
        item.contains("-> :: core :: result :: Result < :: gust :: capture :: Captured < Self > , :: gust :: capture :: Error << __GustD as :: gust :: Device > :: Error > , >"),
        "{item}"
    );
    assert!(
        item.contains("where __GustD : :: gust :: capture :: CaptureDevice , {"),
        "{item}"
    );
    // The twin itself is not changed by it: it stays one impl under one gate.
    assert_eq!(text.matches("impl :: gust :: Run <").count(), 1, "{text}");
}

#[test]
fn every_resource_is_copied_in_parameter_order_with_the_element_type_written() {
    let item = capture_item(&expansion_text(PARTICLES, Profile::Other), "step");
    // Copy: the uniform first (index 0), then the two buffers; each names its parameter for the error messages.
    let copies = [
        ":: gust :: capture :: capture_uniform :: < __GustD , Params > (__gust_device , & __gust_bindings [0usize] , \"params\" , & __gust_cpu ,) ?",
        ":: gust :: capture :: capture_storage :: < __GustD , Particle > (__gust_device , & __gust_bindings [1usize] , \"input\" , :: gust :: Access :: ReadOnly , & __gust_cpu ,) ?",
        ":: gust :: capture :: capture_storage :: < __GustD , Particle > (__gust_device , & __gust_bindings [2usize] , \"out\" , :: gust :: Access :: ReadWrite , & __gust_cpu ,) ?",
    ];
    let mut at = 0;
    for copy in copies {
        let found = item[at..]
            .find(copy)
            .unwrap_or_else(|| panic!("{copy}\n{item}"));
        at += found + copy.len();
    }
    // Rebind for the replay: read-only stays read-only, read-write is writable, the uniform is by copy.
    for rebind in [
        ":: gust :: capture :: bind_uniform :: < Params > (& __gust_slots [0usize]) ?",
        ":: gust :: capture :: bind_storage :: < Particle > (& __gust_slots [1usize] , :: gust :: Access :: ReadOnly) ?",
        ":: gust :: capture :: bind_storage :: < Particle > (& __gust_slots [2usize] , :: gust :: Access :: ReadWrite) ?",
    ] {
        assert!(item.contains(rebind), "{rebind}\n{item}");
    }
}

#[test]
fn the_replay_runs_the_shadow_body_for_each_id_it_is_given() {
    let item = capture_item(&expansion_text(PARTICLES, Profile::Other), "step");
    assert!(
        item.contains(":: gust :: cpu :: validate (__gust_cpu , & __gust_call) ?"),
        "{item}"
    );
    assert!(
        item.contains("let [params , input , out] = * __gust_call . bindings () ;"),
        "{item}"
    );
    // The body is called as the twin calls it, `id` where it is written (last here), once per id.
    let call = "for __gust_id in __gust_ids { __gpu_typecheck_step (";
    let at = item.find(call).unwrap_or_else(|| panic!("{call}\n{item}"));
    let rest = &item[at..];
    let from_params = rest
        .find(":: gust :: Uniform (params)")
        .expect("uniform argument");
    let from_input = rest.find(":: gust :: Storage :: new (& input)").unwrap();
    let from_out = rest
        .find(":: gust :: StorageMut :: new (& mut out)")
        .unwrap();
    let from_id = rest
        .find("< SV_DispatchThreadID as :: gust :: cpu :: FromGlobalId > :: from_global_id (__gust_id)")
        .unwrap();
    assert!(from_params < from_input && from_input < from_out && from_out < from_id);
    // The result owns the copies and the replay function.
    assert!(
        item.contains(
            ":: gust :: capture :: Captured :: new (__gust_cpu , __gust_call . extent () , __gust_slots , __gust_replay ,)"
        ),
        "{item}"
    );
}

#[test]
fn the_call_is_checked_by_the_device_before_anything_is_copied() {
    let item = capture_item(&expansion_text(PARTICLES, Profile::Other), "step");
    let check = ":: gust :: capture :: Error :: Call) ? ;";
    let checked = item
        .find(check)
        .unwrap_or_else(|| panic!("{check}\n{item}"));
    assert!(
        item[..checked].contains("__gust_device . check_call (__gust_call . kernel () , __gust_call . extent () , __gust_call . bindings () ,)"),
        "{item}"
    );
    let first_copy = item.find("capture_uniform :: <").unwrap();
    assert!(checked < first_copy, "{item}");
}

/// The text of the loop of a harness: from the destructuring of the bindings to the end of the `for`.
fn skeleton(item: &str) -> &str {
    let start = item
        .find("let [params , input , out] = * __gust_call . bindings () ;")
        .unwrap_or_else(|| panic!("{item}"));
    let rest = &item[start..];
    let for_at = rest.find("for __gust_id in").unwrap();
    let end = for_at + rest[for_at..].find(") ; }").unwrap() + ") ; }".len();
    &rest[..end]
}

#[test]
fn the_replay_is_the_twins_own_loop_over_the_ids_it_is_given() {
    // Pins the replay's skeleton (acquisitions, arguments, the shadow-body call) against drift from the twin's
    // `submit_call`: the same text, but for the ids.
    let text = expansion_text(PARTICLES, Profile::Other);
    let twin_start = text.find("impl :: gust :: Run <").expect("the twin");
    let twin = skeleton(&text[twin_start..]);
    let item = capture_item(&text, "step");
    let replay = skeleton(&item);
    let walk = ":: gust :: cpu :: invocations (__gust_call . kernel () , __gust_call . extent ())";
    assert!(twin.contains(walk), "{twin}");
    assert_eq!(twin.replace(walk, "__gust_ids"), replay);
}

#[test]
fn a_kernel_without_a_twin_gets_a_capture_that_says_so() {
    let source = "mod vector_add { \
        #[kernel(cpu = \"unsupported\")] \
        pub fn add(id: SV_DispatchThreadID, a: StructuredBuffer<float>, mut out: RWStructuredBuffer<float>) { \
            let i = id.x; if i < out.len() { out[i] = a[i]; } \
        } \
    }";
    let text = expansion_text(source, Profile::Other);
    let item = capture_item(&text, "add");
    assert!(
        item.contains("pub fn capture < 'r , __GustD > (__gust_device : & __GustD , __gust_call : & :: gust :: Call < 'r , __GustD , Self , 2 > ,)"),
        "{item}"
    );
    assert!(
        item.contains(":: core :: result :: Result :: Err (:: gust :: capture :: Error :: NoTwin { kernel : \"add\" , })"),
        "{item}"
    );
    // No copy and no replay: nothing to rebind into.
    assert!(!item.contains("capture_storage"), "{item}");
    assert!(!item.contains("__gust_replay"), "{item}");
    assert!(item.contains("no CPU twin"), "{item}");
}

#[test]
fn a_kernel_with_no_portable_body_gets_the_no_twin_capture_too() {
    let source = "mod m { \
        #[kernel(workgroup_size(256, 1, 1), cpu = \"unsupported\", requires(subgroups))] \
        pub fn lanes(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) { \
            let i = id.x; if i < out.len() { out[i] = 1u32; } \
        } \
    }";
    // The crate declares only a profile with the capability, so the kernel is accepted; none of its bodies runs without
    // it, so it has no twin.
    let directory = std::env::temp_dir().join(format!("gust-g2cap-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(
        directory.join("Cargo.toml"),
        "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n\
         [package.metadata.gust]\nprofiles = [\"native+subgroups\"]\n",
    )
    .unwrap();
    let environment = Environment {
        manifest_dir: Some(directory.clone()),
        export_dir: None,
        package: Some("demo".to_owned()),
        reported_file: None,
    };
    let profile = Profile::parse("dev+subgroups").expect("a profile");
    let module = crate::diagnostics::testing::parse_str(source).unwrap();
    let expanded = expand_in(module, &environment, profile);
    let _ = std::fs::remove_dir_all(&directory);
    let text = expanded.unwrap().to_string();
    let item = capture_item(&text, "lanes");
    assert!(
        item.contains(":: gust :: capture :: Error :: NoTwin { kernel : \"lanes\" , }"),
        "{item}"
    );
    assert!(!item.contains("__gust_replay"), "{item}");
}

#[test]
fn a_texture_kernel_gets_a_capture_that_says_it_is_not_supported_yet() {
    let source = "mod m { #[kernel(workgroup_size(8, 8, 1))] pub fn run(id: SV_DispatchThreadID, \
                  a: StructuredBuffer<float>, mut out: WriteTexture2D<Rgba32Float>) { \
                  let w = out.width(); if id.x < w { out.store(id.x, id.y, Vec4::splat(a[0])); } } }";
    let text = expansion_text(source, Profile::Other);
    let item = capture_item(&text, "run");
    assert!(
        item.contains(":: gust :: capture :: Error :: Unsupported { kernel : \"run\" , parameter : \"out\" , what : \"a texture\" , }"),
        "{item}"
    );
    assert!(!item.contains("capture_storage"), "{item}");
}

#[test]
fn the_cpu_profile_has_no_capture() {
    let text = expansion_text(PARTICLES, Profile::Cpu);
    assert!(!text.contains("capture"), "{text}");
}
