//! Test data written by hand from the record schema (see `record.rs`); no macro and no compiler produced it.
//!
//! Phase 2 of slice m4.d compares the reader with the macro's own committed golden (m4.c's `sample.kernel.json`).
//! The key and both hashes below were computed outside this code: the key with a BigInt FNV-1a 64 script in node
//! over `entry_point`, a zero byte and the text, the SHA-256 with `sha256sum` over the same text in a file.

/// The clean Slang text of the sample kernel (LF line ends, 311 bytes).
pub(crate) const SAMPLE_SLANG: &str = concat!(
    "struct Pair\n",
    "{\n",
    "    uint a;\n",
    "    float b;\n",
    "};\n",
    "\n",
    "[[vk::binding(0, 0)]]\n",
    "StructuredBuffer<Pair> pairs;\n",
    "\n",
    "[[vk::binding(1, 0)]]\n",
    "RWStructuredBuffer<float> out;\n",
    "\n",
    "[shader(\"compute\")]\n",
    "[numthreads(64, 1, 1)]\n",
    "void gpu_demo_run(uint3 id : SV_DispatchThreadID)\n",
    "{\n",
    "    uint i = id.x;\n",
    "    out[i] = (float)pairs[i].a + pairs[i].b;\n",
    "}\n"
);

/// FNV-1a 64 over `gpu_demo_run`, a zero byte and [`SAMPLE_SLANG`].
pub(crate) const SAMPLE_KEY: &str = "022c52d675787198";

/// SHA-256 of [`SAMPLE_SLANG`].
pub(crate) const SAMPLE_SLANG_SHA256: &str =
    "5110bc65c7d5f23fa588a08e69b24c6156cfd1ddf4a280766209bee9781298e1";

/// The kernel record of the sample kernel, in the form the macro prints it (schema 1, sorted keys, two-space
/// indentation, LF).
pub(crate) const SAMPLE_RECORD_JSON: &str = concat!(
    "{\n",
    "  \"entry_point\": \"gpu_demo_run\",\n",
    "  \"key\": \"022c52d675787198\",\n",
    "  \"module\": \"demo\",\n",
    "  \"name\": \"run\",\n",
    "  \"parameters\": [\n",
    "    {\n",
    "      \"access\": \"not_applicable\",\n",
    "      \"binding\": null,\n",
    "      \"kind\": \"invocation\",\n",
    "      \"layout\": null,\n",
    "      \"name\": \"id\",\n",
    "      \"rust_type\": \"SV_DispatchThreadID\"\n",
    "    },\n",
    "    {\n",
    "      \"access\": \"read_only\",\n",
    "      \"binding\": {\n",
    "        \"binding\": 0,\n",
    "        \"group\": 0\n",
    "      },\n",
    "      \"kind\": \"storage\",\n",
    "      \"layout\": {\n",
    "        \"element\": {\n",
    "          \"alignment\": 4,\n",
    "          \"fields\": [\n",
    "            {\n",
    "              \"name\": \"a\",\n",
    "              \"offset\": 0,\n",
    "              \"type\": {\n",
    "                \"alignment\": 4,\n",
    "                \"kind\": \"scalar\",\n",
    "                \"name\": \"u32\",\n",
    "                \"scalar\": \"u32\",\n",
    "                \"size\": 4,\n",
    "                \"stride\": 4\n",
    "              }\n",
    "            },\n",
    "            {\n",
    "              \"name\": \"b\",\n",
    "              \"offset\": 4,\n",
    "              \"type\": {\n",
    "                \"alignment\": 4,\n",
    "                \"kind\": \"scalar\",\n",
    "                \"name\": \"f32\",\n",
    "                \"scalar\": \"f32\",\n",
    "                \"size\": 4,\n",
    "                \"stride\": 4\n",
    "              }\n",
    "            }\n",
    "          ],\n",
    "          \"kind\": \"struct\",\n",
    "          \"name\": \"Pair\",\n",
    "          \"size\": 8,\n",
    "          \"stride\": 8\n",
    "        },\n",
    "        \"element_stride\": 8,\n",
    "        \"rules\": \"storage_v1\"\n",
    "      },\n",
    "      \"name\": \"pairs\",\n",
    "      \"rust_type\": \"StructuredBuffer < Pair >\"\n",
    "    },\n",
    "    {\n",
    "      \"access\": \"read_write\",\n",
    "      \"binding\": {\n",
    "        \"binding\": 1,\n",
    "        \"group\": 0\n",
    "      },\n",
    "      \"kind\": \"storage\",\n",
    "      \"layout\": {\n",
    "        \"element\": {\n",
    "          \"alignment\": 4,\n",
    "          \"kind\": \"scalar\",\n",
    "          \"name\": \"f32\",\n",
    "          \"scalar\": \"f32\",\n",
    "          \"size\": 4,\n",
    "          \"stride\": 4\n",
    "        },\n",
    "        \"element_stride\": 4,\n",
    "        \"rules\": \"storage_v1\"\n",
    "      },\n",
    "      \"name\": \"out\",\n",
    "      \"rust_type\": \"RWStructuredBuffer < float >\"\n",
    "    }\n",
    "  ],\n",
    "  \"schema\": 1,\n",
    "  \"source\": {\n",
    "    \"file\": \"src/lib.rs\",\n",
    "    \"lines\": [\n",
    "      [3, 9, 12],\n",
    "      [5, 5, 14]\n",
    "    ]\n",
    "  },\n",
    "  \"workgroup_size\": [64, 1, 1]\n",
    "}\n"
);

/// The sample kernel's record as a `par_for!` call writes it (schema 2, R1.1 M5): the same keys, no `source`. The
/// record does not depend on where its body stands, so it is derived from [`SAMPLE_RECORD_JSON`] by dropping the
/// source block and raising the schema.
pub(crate) fn site_record_json() -> String {
    let source = concat!(
        "  \"source\": {\n",
        "    \"file\": \"src/lib.rs\",\n",
        "    \"lines\": [\n",
        "      [3, 9, 12],\n",
        "      [5, 5, 14]\n",
        "    ]\n",
        "  },\n"
    );
    assert!(SAMPLE_RECORD_JSON.contains(source));
    SAMPLE_RECORD_JSON
        .replace(source, "")
        .replace("\"schema\": 1", "\"schema\": 2")
}

/// The sidecar names of two call sites of the sample kernel, as the macro names them (`<key>.<site16>.site.json`,
/// `site16` = FNV-1a 64 over the relative file, a zero byte and the decimal call line; computed outside this code
/// with a script): `src/lib.rs` at line 12 and `tests/other.rs` at line 40. The second sorts first.
pub(crate) const SITE_LIB_NAME: &str = "022c52d675787198.6b7d03da5734408d.site.json";
pub(crate) const SITE_OTHER_NAME: &str = "022c52d675787198.440c6d8bbdeab608.site.json";

/// The sidecar of the call in `src/lib.rs`.
pub(crate) const SITE_LIB_JSON: &str = concat!(
    "{\n",
    "  \"file\": \"src/lib.rs\",\n",
    "  \"key\": \"022c52d675787198\",\n",
    "  \"lines\": [\n",
    "    [3, 9, 12],\n",
    "    [5, 5, 14]\n",
    "  ],\n",
    "  \"schema\": 2\n",
    "}\n"
);

/// The sidecar of the call in `tests/other.rs`: the same body at another place.
pub(crate) const SITE_OTHER_JSON: &str = concat!(
    "{\n",
    "  \"file\": \"tests/other.rs\",\n",
    "  \"key\": \"022c52d675787198\",\n",
    "  \"lines\": [\n",
    "    [3, 9, 40],\n",
    "    [5, 5, 42]\n",
    "  ],\n",
    "  \"schema\": 2\n",
    "}\n"
);

/// The vector-add kernel's WGSL and the raw reflection JSON the helper wrote for it (spike S1, kept verbatim in gust's
/// fixtures; the key is the one its macro-made descriptor has).
pub(crate) const VECTOR_ADD_KEY: &str = "52e85c5f03f72272";
pub(crate) const VECTOR_ADD_WGSL: &[u8] =
    include_bytes!("../../gust/tests/fixtures/precompiled/52e85c5f03f72272.wgsl");
pub(crate) const VECTOR_ADD_REFLECTION: &str =
    include_str!("../../gust/tests/fixtures/precompiled/52e85c5f03f72272.wgsl.reflection.json");

/// The goldens of a crate with variants (slice m8.b2, D139): what the macros and `cargo gust build` made for two
/// kernels with a variant each, kept under `testdata/variant/`.
///
/// They come from a real `cargo gust build` of a scratch crate whose `src/lib.rs` held the two modules below, with the
/// pinned Slang: the macro's records and clean Slang text of all four bodies (`<key>.kernel.json`, `<key>.slang`), the
/// WGSL and reflection JSON the helper wrote for each, and the `manifest.json` (schema 3) the build wrote. The tests
/// check them against the macro as it is today (the descriptor of the two bodies a default build selects), against the
/// reader, and against a plan built from the records alone.
pub(crate) mod variant {
    use gust::{Format, KernelDescriptor, artifact::content_key, capability::Capability};

    use crate::{
        export::{
            Compiled, Compiler, HelperCompiler, plan, variant_lists, variant_problems,
            verify_stored,
        },
        manifest::{Manifest, VARIANT_SCHEMA},
        record::Record,
    };

    // The same text as the scratch crate's `src/lib.rs`. A default build selects `double` (the kernel's own, portable
    // body) of the first and `total_portable` (the variant that needs nothing) of the second, so these two modules
    // give two of the four descriptors; the records of the other two are checked against their own text.
    #[gust::gpu]
    mod variant_probe {
        #[kernel(workgroup_size(64, 1, 1))]
        pub fn double(
            id: SV_DispatchThreadID,
            data: StructuredBuffer<u32>,
            mut out: RWStructuredBuffer<u32>,
        ) {
            if id.x < out.len() {
                out[id.x] = data[id.x] * 2u32;
            }
        }

        #[variant(of = double, requires(subgroups))]
        pub fn double_subgroups(
            id: SV_DispatchThreadID,
            data: StructuredBuffer<u32>,
            mut out: RWStructuredBuffer<u32>,
        ) {
            if id.x < out.len() {
                out[id.x] = data[id.x] + data[id.x];
            }
        }
    }

    #[gust::gpu]
    mod gated_probe {
        #[kernel(workgroup_size(64, 1, 1), requires(subgroups))]
        pub fn total(
            id: SV_DispatchThreadID,
            data: StructuredBuffer<u32>,
            mut out: RWStructuredBuffer<u32>,
        ) {
            if id.x < out.len() {
                out[id.x] = data[id.x] + 1u32;
            }
        }

        #[variant(of = total, requires())]
        pub fn total_portable(
            id: SV_DispatchThreadID,
            data: StructuredBuffer<u32>,
            mut out: RWStructuredBuffer<u32>,
        ) {
            if id.x < out.len() {
                out[id.x] = 1u32 + data[id.x];
            }
        }
    }

    /// The manifest the build wrote (schema 3).
    pub(crate) const MANIFEST: &str = include_str!("testdata/variant/manifest.json");

    /// `variant_probe::double`, the kernel's own portable body: schema 1, as before M8.
    pub(crate) const DOUBLE_KEY: &str = "83b6f7fcc126e530";
    /// `variant_probe::double_subgroups`, a variant that needs subgroups: schema 4.
    pub(crate) const DOUBLE_SUBGROUPS_KEY: &str = "08adc71357b5966e";
    /// `gated_probe::total`, an own body that needs subgroups (`variant` ""): schema 4.
    pub(crate) const TOTAL_KEY: &str = "0de18f77cf4dfd53";
    /// `gated_probe::total_portable`, the portable variant with an empty set: schema 4.
    pub(crate) const TOTAL_PORTABLE_KEY: &str = "4714627abe9c3d3b";

    /// One stored body.
    pub(crate) struct Body {
        pub(crate) key: &'static str,
        pub(crate) entry_point: &'static str,
        /// The kernel's name, which every body of it carries.
        pub(crate) kernel: &'static str,
        /// The record's `schema`.
        pub(crate) schema: u32,
        pub(crate) variant: &'static str,
        pub(crate) requires: &'static [Capability],
        pub(crate) record_json: &'static str,
        pub(crate) slang: &'static str,
        pub(crate) wgsl: &'static [u8],
        pub(crate) reflection: &'static str,
        /// The descriptor a default build of the macro makes for it, when it is the body that build selects.
        pub(crate) selected: Option<&'static KernelDescriptor>,
    }

    macro_rules! stored {
        ($key:literal, $entry_point:literal, $kernel:literal, $schema:literal, $variant:literal,
         $requires:expr, $selected:expr) => {
            Body {
                key: $key,
                entry_point: $entry_point,
                kernel: $kernel,
                schema: $schema,
                variant: $variant,
                requires: $requires,
                record_json: include_str!(concat!("testdata/variant/", $key, ".kernel.json")),
                slang: include_str!(concat!("testdata/variant/", $key, ".slang")),
                wgsl: include_bytes!(concat!("testdata/variant/", $key, ".wgsl")),
                reflection: include_str!(concat!(
                    "testdata/variant/",
                    $key,
                    ".wgsl.reflection.json"
                )),
                selected: $selected,
            }
        };
    }

    pub(crate) const BODIES: &[Body] = &[
        stored!(
            "83b6f7fcc126e530",
            "gpu_variant_probe_double",
            "double",
            1,
            "",
            &[],
            Some(&variant_probe::double::DESCRIPTOR)
        ),
        stored!(
            "08adc71357b5966e",
            "gpu_variant_probe_double_subgroups",
            "double",
            4,
            "double_subgroups",
            &[Capability::Subgroups],
            None
        ),
        stored!(
            "0de18f77cf4dfd53",
            "gpu_gated_probe_total",
            "total",
            4,
            "",
            &[Capability::Subgroups],
            None
        ),
        stored!(
            "4714627abe9c3d3b",
            "gpu_gated_probe_total_portable",
            "total",
            4,
            "total_portable",
            &[],
            Some(&gated_probe::total::DESCRIPTOR)
        ),
    ];

    impl Body {
        pub(crate) fn record(&self) -> Record {
            Record::parse(self.record_json, self.slang).unwrap()
        }
    }

    pub(crate) fn records() -> Vec<Record> {
        BODIES.iter().map(Body::record).collect()
    }

    /// `HelperCompiler::compile` with the stored bytes in place of the helper's run: the same verification as a program
    /// does, the same `Compiled`.
    pub(crate) struct ReplayCompiler;

    impl Compiler for ReplayCompiler {
        fn compile(
            &self,
            record: &Record,
            _kernel: &KernelDescriptor,
            format: Format,
        ) -> Result<Compiled, String> {
            assert_eq!(format, Format::Wgsl);
            let stored = BODIES
                .iter()
                .find(|body| body.key == record.key)
                .ok_or("no stored artifact for this key")?;
            let compiler = verify_stored(record, format, stored.wgsl, stored.reflection)?;
            Ok(Compiled {
                bytes: stored.wgsl.to_vec(),
                reflection_json: stored.reflection.to_owned(),
                compiler,
                validated: None,
                warnings: Vec::new(),
            })
        }
    }

    #[test]
    fn the_stored_records_are_the_macros_and_carry_the_facts_of_their_bodies() {
        for body in BODIES {
            assert_eq!(
                format!("{:016x}", content_key(body.entry_point, body.slang)),
                body.key,
                "{}",
                body.entry_point
            );
            let schema = format!("\"schema\": {},", body.schema);
            assert!(body.record_json.contains(&schema), "{}", body.entry_point);
            assert_eq!(
                body.record_json.contains("\"variant\""),
                body.schema == 4,
                "{}",
                body.entry_point
            );
            let record = body.record();
            assert_eq!(record.key, body.key);
            assert_eq!(record.entry_point, body.entry_point);
            assert_eq!(record.variant, body.variant, "{}", body.entry_point);
            assert_eq!(record.requires, body.requires, "{}", body.entry_point);
            // Every body of a kernel carries the kernel's name; the variant's function name is `variant`.
            assert_eq!(record.name, body.kernel, "{}", body.entry_point);
            // The descriptor of the record carries the facts too.
            let descriptor = record.descriptor();
            assert_eq!(descriptor.variant, body.variant);
            assert_eq!(descriptor.requires, body.requires);
        }
    }

    #[test]
    fn the_macro_makes_today_what_the_stored_selected_bodies_are() {
        let mut checked = 0;
        for body in BODIES {
            let Some(kernel) = body.selected else {
                continue;
            };
            checked += 1;
            assert_eq!(kernel.entry_point, body.entry_point);
            assert_eq!(
                kernel.slang_source, body.slang,
                "{}: the macro drifted",
                body.entry_point
            );
            assert_eq!(kernel.variant, body.variant, "{}", body.entry_point);
            assert_eq!(kernel.requires, body.requires, "{}", body.entry_point);
            assert_eq!(
                format!(
                    "{:016x}",
                    content_key(kernel.entry_point, kernel.slang_source)
                ),
                body.key
            );
            // The record, read back, gives the descriptor the macro made.
            let from_record = body.record().descriptor();
            assert_eq!(from_record.entry_point, kernel.entry_point);
            assert_eq!(from_record.workgroup_size, kernel.workgroup_size);
            assert_eq!(from_record.parameters.len(), kernel.parameters.len());
        }
        assert_eq!(checked, 2);
    }

    #[test]
    fn the_plan_of_the_stored_records_is_the_stored_manifest_byte_for_byte() {
        let mut records = records();
        // Whatever order they are collected in.
        records.reverse();
        let planned = plan(records, &[Format::Wgsl], &ReplayCompiler).unwrap();
        assert_eq!(planned.manifest.schema(), VARIANT_SCHEMA);
        assert_eq!(planned.manifest.render(), MANIFEST);
        assert_eq!(planned.manifest_file().bytes, MANIFEST.as_bytes());
        // The files are the stored bytes, two for each of the four bodies.
        assert_eq!(planned.files.len(), 8);
        for body in BODIES {
            let wgsl = planned
                .files
                .iter()
                .find(|file| file.name == format!("{}.wgsl", body.key))
                .unwrap();
            assert_eq!(wgsl.bytes, body.wgsl);
            let reflection = planned
                .files
                .iter()
                .find(|file| file.name == format!("{}.wgsl.reflection.json", body.key))
                .unwrap();
            assert_eq!(reflection.bytes, body.reflection.as_bytes());
        }
        assert_eq!(planned.manifest.slang.as_deref(), Some("2026.20"));
    }

    #[test]
    fn the_stored_manifest_reads_back_and_is_what_the_sources_give() {
        let manifest = Manifest::parse(MANIFEST).unwrap();
        assert_eq!(
            manifest.render(),
            MANIFEST,
            "the stored manifest is not canonical"
        );
        assert_eq!(manifest.kernels.len(), 4);
        assert_eq!(
            variant_problems(&records(), &manifest),
            Vec::<String>::new()
        );
        // The two lists: each kernel's own body carries its kernel's bodies, sorted by key.
        let lists = variant_lists(&records()).unwrap();
        let mut owners: Vec<&str> = lists.iter().map(|(owner, _)| owner.as_str()).collect();
        owners.sort_unstable();
        assert_eq!(owners, [TOTAL_KEY, DOUBLE_KEY]);
        for (owner, bodies) in &lists {
            assert_eq!(&manifest.entry(owner).unwrap().variants, bodies);
        }
        let double = &manifest.entry(DOUBLE_KEY).unwrap().variants;
        assert_eq!(double.len(), 2);
        assert_eq!(double[0].key, DOUBLE_SUBGROUPS_KEY);
        assert_eq!(double[0].requires, [Capability::Subgroups]);
        assert_eq!(double[0].variant, "double_subgroups");
        assert_eq!(double[1].key, DOUBLE_KEY);
        assert!(double[1].requires.is_empty() && double[1].variant.is_empty());
        // `total` needs subgroups itself; its portable variant needs nothing.
        let total = &manifest.entry(TOTAL_KEY).unwrap().variants;
        assert_eq!(total.len(), 2);
        assert_eq!(total[0].key, TOTAL_KEY);
        assert_eq!(total[0].requires, [Capability::Subgroups]);
        assert_eq!(total[1].key, TOTAL_PORTABLE_KEY);
        assert_eq!(total[1].variant, "total_portable");
    }

    #[test]
    fn without_the_variants_the_same_kernel_gives_a_schema_1_manifest() {
        // The kernel's own body of the first module has no facts of its own: alone it is a plain kernel.
        let alone = vec![BODIES[0].record()];
        let planned = plan(alone, &[Format::Wgsl], &ReplayCompiler).unwrap();
        assert_eq!(planned.manifest.schema(), crate::manifest::SCHEMA);
        assert!(!planned.manifest.render().contains("variants"));
        assert!(planned.manifest.render().contains("\"schema\": 1,"));
        assert!(planned.manifest.kernels[0].variants.is_empty());
    }

    #[test]
    fn the_helper_builds_the_stored_artifacts() {
        // The real compile step (`HelperCompiler`: the helper, the whole cross-check, the run-time verification) on the
        // stored records gives the stored bytes.
        let compiler = HelperCompiler {
            helper: gust::reflect::native_compiler_path(),
            spirv_val: None,
        };
        for body in BODIES {
            let record = body.record();
            let compiled = compiler
                .compile(&record, &record.descriptor(), Format::Wgsl)
                .unwrap();
            assert_eq!(compiled.bytes, body.wgsl, "{}", body.entry_point);
            assert_eq!(
                compiled.reflection_json, body.reflection,
                "{}",
                body.entry_point
            );
            assert_eq!(compiled.compiler, "2026.20");
        }
    }
}

/// What the helper reports for the `pairs` buffer of the sample kernel: the same layout the record spells out.
pub(crate) const PAIRS_REFLECTION: &str = r#"{"parameters":[{"name":"pairs","elementStride":8,"type":{"kind":"resource","baseShape":"structuredBuffer","access":"read","resultType":{"size":8,"alignment":4,"stride":8,"name":"Pair","kind":"struct","fields":[{"name":"a","binding":{"kind":"uniform","offset":0,"size":4},"type":{"size":4,"alignment":4,"stride":4,"name":"uint","kind":"scalar","scalarType":"uint32"}},{"name":"b","binding":{"kind":"uniform","offset":4,"size":4},"type":{"size":4,"alignment":4,"stride":4,"name":"float","kind":"scalar","scalarType":"float32"}}]}}}]}"#;
