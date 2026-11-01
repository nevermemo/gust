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

/// The vector-add kernel's WGSL and the raw reflection JSON the helper wrote for it (spike S1, kept verbatim in gust's
/// fixtures; the key is the one its macro-made descriptor has).
pub(crate) const VECTOR_ADD_KEY: &str = "52e85c5f03f72272";
pub(crate) const VECTOR_ADD_WGSL: &[u8] =
    include_bytes!("../../gust/tests/fixtures/precompiled/52e85c5f03f72272.wgsl");
pub(crate) const VECTOR_ADD_REFLECTION: &str =
    include_str!("../../gust/tests/fixtures/precompiled/52e85c5f03f72272.wgsl.reflection.json");

/// What the helper reports for the `pairs` buffer of the sample kernel: the same layout the record spells out.
pub(crate) const PAIRS_REFLECTION: &str = r#"{"parameters":[{"name":"pairs","elementStride":8,"type":{"kind":"resource","baseShape":"structuredBuffer","access":"read","resultType":{"size":8,"alignment":4,"stride":8,"name":"Pair","kind":"struct","fields":[{"name":"a","binding":{"kind":"uniform","offset":0,"size":4},"type":{"size":4,"alignment":4,"stride":4,"name":"uint","kind":"scalar","scalarType":"uint32"}},{"name":"b","binding":{"kind":"uniform","offset":4,"size":4},"type":{"size":4,"alignment":4,"stride":4,"name":"float","kind":"scalar","scalarType":"float32"}}]}}}]}"#;
