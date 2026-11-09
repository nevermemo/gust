// The source of truth of gust-slang-reflect, the native reflection helper. cargo-gust ships a copy as
// crates/cargo-gust/assets/slang-layout.cpp (a cargo-gust test fails when the two differ); after an edit copy this file
// over it. Build the helper with `cargo gust slang build-helper`, or `cargo xtask build-slang-reflect` in this repository.
#include <slang.h>
#include <slang-com-ptr.h>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

using Slang::ComPtr;

static void diagnostics(slang::IBlob* blob)
{
    if (blob && blob->getBufferSize())
        std::cerr.write(static_cast<const char*>(blob->getBufferPointer()), blob->getBufferSize());
}

static void check(SlangResult result, const char* operation, slang::IBlob* blob = nullptr)
{
    diagnostics(blob);
    if (SLANG_FAILED(result)) throw std::runtime_error(operation);
}

static size_t known(size_t value, const char* what)
{
    if (value == SLANG_UNKNOWN_SIZE) throw std::runtime_error(std::string(what) + " is unknown");
    if (value == SLANG_UNBOUNDED_SIZE) throw std::runtime_error(std::string(what) + " is unbounded");
    if (value > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error(std::string(what) + " exceeds u32");
    return value;
}

static std::string json_string(const char* text)
{
    if (!text) throw std::runtime_error("missing reflected name");
    std::ostringstream output;
    output << '"';
    for (unsigned char character : std::string(text))
    {
        if (character == '"' || character == '\\') output << '\\' << character;
        else if (character < 0x20)
            output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(character);
        else output << character;
    }
    output << '"';
    return output.str();
}

// The matrix layout is pinned, not left to the API default (row-major in the native API, column-major on the slangc
// command line: M6 spike V1). The CLI path passes -matrix-layout-column-major for the same reason.
static const SlangMatrixLayoutMode kMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;

static const char* matrix_layout_name(SlangMatrixLayoutMode mode)
{
    switch (mode)
    {
    case SLANG_MATRIX_LAYOUT_COLUMN_MAJOR: return "column";
    case SLANG_MATRIX_LAYOUT_ROW_MAJOR: return "row";
    default: throw std::runtime_error("unknown matrix layout mode");
    }
}

static const char* scalar_name(slang::TypeReflection::ScalarType scalar)
{
    switch (scalar)
    {
    case slang::TypeReflection::ScalarType::Float32: return "float32";
    case slang::TypeReflection::ScalarType::Int32: return "int32";
    case slang::TypeReflection::ScalarType::UInt32: return "uint32";
    default: throw std::runtime_error("unsupported scalar type; only 32-bit scalars are supported");
    }
}

// Schema 1 describes 32-bit scalars and structs of them in structured buffers. A vector, a matrix or a constant buffer
// anywhere in the program makes the document schema 2: a top-level matrixLayout and the new type kinds. A texture, a
// sampler or a stage other than compute makes it schema 3 (mx.t0a): schema 2's grammar (matrixLayout included) plus a
// top-level "stage", texture and sampler bindings and, for a fragment entry, "targets" and no "workgroupSize". Nothing
// else writes schema 3, so every document a compute kernel over buffers produced before stays byte-identical. An array
// anywhere in a buffer element makes it schema 4 (m9.r2h): schema 3's grammar (stage, textures, samplers, targets) plus the
// type kind "array" (elementCount, elementStride, elementType). Nothing else writes schema 4 either.
struct SchemaUse
{
    bool newKinds = false;
    bool shapes = false;
    bool arrays = false;
};

// The pipeline stage the caller expects the entry point to have (the optional seventh argument; compute by default).
enum class Stage
{
    Compute,
    Fragment
};

static const char* stage_name(Stage stage)
{
    return stage == Stage::Compute ? "compute" : "fragment";
}

static SlangStage stage_slang(Stage stage)
{
    return stage == Stage::Compute ? SLANG_STAGE_COMPUTE : SLANG_STAGE_FRAGMENT;
}

static std::string upper_case(const char* text)
{
    std::string result = text;
    for (auto& character : result)
        character = static_cast<char>(std::toupper(static_cast<unsigned char>(character)));
    return result;
}

static std::string fingerprint(const void* data, size_t size)
{
    uint64_t hash = UINT64_C(14695981039346656037);
    const auto bytes = static_cast<const unsigned char*>(data);
    for (size_t index = 0; index < size; ++index)
    {
        hash ^= bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    std::ostringstream output;
    output << std::hex << std::setw(16) << std::setfill('0') << hash;
    return output.str();
}

static std::string type_json(slang::TypeLayoutReflection* type, SchemaUse& use, unsigned depth = 0)
{
    if (!type || depth > 32) throw std::runtime_error("missing or excessively nested element layout");
    const auto size = known(type->getSize(), "size");
    auto stride = known(type->getStride(), "stride");
    const auto alignment = type->getAlignment();
    if (size == 0 || stride == 0 || alignment <= 0)
        throw std::runtime_error("missing byte layout");
    if (type->getKind() == slang::TypeReflection::Kind::Matrix)
    {
        // The "stride" of a matrix in the document is its column stride (its row stride in a row-major matrix). The
        // API's stride is the array stride of the whole matrix, which a structured buffer reports as elementStride.
        const auto major = type->getMatrixLayoutMode() == SLANG_MATRIX_LAYOUT_ROW_MAJOR ? type->getRowCount()
                                                                                         : type->getColumnCount();
        if (major == 0 || size % major) throw std::runtime_error("matrix size is not a whole number of columns");
        stride = size / major;
    }
    std::ostringstream output;
    output << "{\"size\":" << size << ",\"alignment\":" << alignment << ",\"stride\":" << stride;
    if (type->getName()) output << ",\"name\":" << json_string(type->getName());
    switch (type->getKind())
    {
    case slang::TypeReflection::Kind::Scalar:
        output << ",\"kind\":\"scalar\",\"scalarType\":" << json_string(scalar_name(type->getScalarType()));
        break;
    case slang::TypeReflection::Kind::Vector:
        use.newKinds = true;
        output << ",\"kind\":\"vector\",\"scalarType\":" << json_string(scalar_name(type->getScalarType()))
               << ",\"elementCount\":" << type->getType()->getElementCount();
        break;
    case slang::TypeReflection::Kind::Matrix:
        use.newKinds = true;
        output << ",\"kind\":\"matrix\",\"scalarType\":" << json_string(scalar_name(type->getScalarType()))
               << ",\"rows\":" << type->getRowCount() << ",\"columns\":" << type->getColumnCount()
               << ",\"matrixLayout\":" << json_string(matrix_layout_name(type->getMatrixLayoutMode()));
        break;
    case slang::TypeReflection::Kind::Array:
    {
        // "stride" above is the stride of this array as an element of something else; "elementStride" is the distance
        // between its own elements (Slang's uniformStride). An unsized array is unbounded and is refused by known().
        use.arrays = true;
        const auto count = known(type->getElementCount(), "array element count");
        const auto elementStride = known(type->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM), "array element stride");
        auto element = type->getElementTypeLayout();
        if (count == 0 || elementStride == 0 || !element) throw std::runtime_error("missing array layout");
        output << ",\"kind\":\"array\",\"elementCount\":" << count << ",\"elementStride\":" << elementStride
               << ",\"elementType\":" << type_json(element, use, depth + 1);
        break;
    }
    case slang::TypeReflection::Kind::Struct:
        if (!type->getFieldCount()) throw std::runtime_error("empty struct layout");
        output << ",\"kind\":\"struct\",\"fields\":[";
        for (unsigned index = 0; index < type->getFieldCount(); ++index)
        {
            auto field = type->getFieldByIndex(index);
            if (!field || !field->getTypeLayout()) throw std::runtime_error("missing field layout");
            if (field->getCategoryCount() != 1 ||
                field->getCategoryByIndex(0) != slang::ParameterCategory::Uniform)
                throw std::runtime_error("unsupported field layout category");
            if (index) output << ',';
            output << "{\"name\":" << json_string(field->getName())
                   << ",\"binding\":{\"kind\":\"uniform\",\"offset\":"
                   << known(field->getOffset(), "field offset") << ",\"size\":"
                   << known(field->getTypeLayout()->getSize(), "field size") << "},\"type\":"
                   << type_json(field->getTypeLayout(), use, depth + 1) << '}';
        }
        output << ']';
        break;
    default:
        throw std::runtime_error(
            "unsupported element type; only 32-bit scalars, vectors, matrices, arrays and structs are supported");
    }
    output << '}';
    return output.str();
}

// The Slang name of a storage-texture format ("r32ui", "rgba32f", ...), from the same table slang.h builds the enum
// from. Only a read-write or write-only texture carries one; a sampled texture reports "unknown" (see below).
static const char* image_format_name(SlangImageFormat format)
{
    switch (format)
    {
#define SLANG_FORMAT(NAME, DESC) \
    case SLANG_IMAGE_FORMAT_##NAME: return #NAME;
#include "slang-image-format-defs.h"
#undef SLANG_FORMAT
    default: throw std::runtime_error("unknown image format");
    }
}

// A type without a byte layout (a texture's result type, a fragment output): kind, scalarType and elementCount spelled
// as type_json does, and no size, alignment or stride.
static std::string value_type_json(slang::TypeReflection* type)
{
    if (!type) throw std::runtime_error("missing texture result or fragment output type");
    std::ostringstream output;
    switch (type->getKind())
    {
    case slang::TypeReflection::Kind::Scalar:
        output << "{\"kind\":\"scalar\",\"scalarType\":" << json_string(scalar_name(type->getScalarType())) << '}';
        break;
    case slang::TypeReflection::Kind::Vector:
        output << "{\"kind\":\"vector\",\"scalarType\":" << json_string(scalar_name(type->getScalarType()))
               << ",\"elementCount\":" << type->getElementCount() << '}';
        break;
    default:
        throw std::runtime_error(
            "unsupported texture result or fragment output type; only 32-bit scalars and vectors are supported");
    }
    return output.str();
}

// The binding object of a texture or sampler: one descriptor slot, as for a buffer.
static std::string slot_binding_json(slang::VariableLayoutReflection* resource, slang::TypeLayoutReflection* type)
{
    const auto category = slang::ParameterCategory::DescriptorTableSlot;
    if (resource->getCategoryCount() != 1 || resource->getCategoryByIndex(0) != category ||
        type->getSize(category) != 1)
        throw std::runtime_error("only individual texture and sampler descriptor bindings are supported");
    std::ostringstream output;
    output << "{\"kind\":\"descriptorTableSlot\",\"count\":1,\"index\":"
           << known(resource->getOffset(category), "binding index") << ",\"space\":"
           << known(resource->getBindingSpace(category), "binding space") << '}';
    return output.str();
}

// A fragment entry's outputs in SV_Target order: {"index":n,"type":{..}} per target. The result is either one
// SV_Target scalar or vector, or a struct whose every field is one; anything else (SV_Depth, a nested struct) is out.
static std::string targets_json(slang::EntryPointReflection* entry)
{
    std::map<unsigned, std::string> targets;
    const auto add = [&targets](slang::VariableLayoutReflection* variable)
    {
        if (!variable || !variable->getSemanticName() || !variable->getTypeLayout() ||
            upper_case(variable->getSemanticName()) != "SV_TARGET")
            throw std::runtime_error("only SV_Target fragment outputs are supported");
        const auto index = static_cast<unsigned>(variable->getSemanticIndex());
        if (!targets.emplace(index, value_type_json(variable->getTypeLayout()->getType())).second)
            throw std::runtime_error("duplicate SV_Target index");
    };
    auto result = entry->getResultVarLayout();
    if (result && result->getTypeLayout())
    {
        auto type = result->getTypeLayout();
        switch (type->getKind())
        {
        case slang::TypeReflection::Kind::None: break;
        case slang::TypeReflection::Kind::Struct:
            for (unsigned index = 0; index < type->getFieldCount(); ++index) add(type->getFieldByIndex(index));
            break;
        default: add(result); break;
        }
    }
    std::ostringstream output;
    output << '[';
    bool first = true;
    for (const auto& target : targets)
    {
        if (!first) output << ',';
        first = false;
        output << "{\"index\":" << target.first << ",\"type\":" << target.second << '}';
    }
    output << ']';
    return output.str();
}

static std::string reflection_json(slang::ProgramLayout* layout, const char* entryName,
                                   const char* target, const char* compiler,
                                   const std::string& source, slang::IBlob* code, Stage stage = Stage::Compute)
{
    if (layout->getEntryPointCount() != 1) throw std::runtime_error("expected exactly one entry point");
    auto entry = layout->getEntryPointByIndex(0);
    if (!entry || !entry->getName() || std::string(entry->getName()) != entryName ||
        entry->getStage() != stage_slang(stage))
        throw std::runtime_error(std::string("entry point identity or ") + stage_name(stage) + " stage mismatch");
    const bool compute = stage == Stage::Compute;
    for (unsigned index = 0; index < entry->getParameterCount(); ++index)
    {
        // A compute entry takes the dispatch thread id; a fragment entry may take its position and nothing else.
        const char* unsupported = compute ? "only dispatch-thread-ID entry-point parameters are supported"
                                          : "only SV_Position entry-point parameters are supported";
        auto parameter = entry->getParameterByIndex(index);
        if (!parameter || !parameter->getSemanticName() || !parameter->getTypeLayout())
            throw std::runtime_error(unsupported);
        const auto semantic = upper_case(parameter->getSemanticName());
        auto type = parameter->getTypeLayout();
        if (semantic != (compute ? "SV_DISPATCHTHREADID" : "SV_POSITION") ||
            type->getKind() != slang::TypeReflection::Kind::Vector ||
            type->getType()->getElementCount() != (compute ? 3u : 4u) ||
            type->getScalarType() !=
                (compute ? slang::TypeReflection::ScalarType::UInt32 : slang::TypeReflection::ScalarType::Float32))
            throw std::runtime_error(unsupported);
        for (unsigned category = 0; category < parameter->getCategoryCount(); ++category)
            if (parameter->getCategoryByIndex(category) != slang::ParameterCategory::VaryingInput)
                throw std::runtime_error("entry-point resource/uniform bindings are unsupported");
    }
    SlangUInt dimensions[3] = {};
    if (compute) entry->getComputeThreadGroupSize(3, dimensions);
    SchemaUse use;
    use.shapes = !compute;
    std::ostringstream parameters;
    for (unsigned index = 0; index < layout->getParameterCount(); ++index)
    {
        auto resource = layout->getParameterByIndex(index);
        if (!resource || !resource->getTypeLayout()) throw std::runtime_error("missing resource layout");
        auto type = resource->getTypeLayout();
        const auto category = slang::ParameterCategory::DescriptorTableSlot;
        if (type->getKind() == slang::TypeReflection::Kind::ConstantBuffer)
        {
            // A constant buffer occupies one descriptor slot; only its element carries a byte layout (Uniform).
            if (resource->getCategoryCount() != 1 || resource->getCategoryByIndex(0) != category ||
                type->getSize(category) != 1)
                throw std::runtime_error("only individual constant-buffer descriptor bindings are supported");
            auto element = type->getElementTypeLayout();
            if (!element) throw std::runtime_error("missing constant-buffer element layout");
            use.newKinds = true;
            const auto elementJson = type_json(element, use);
            if (index) parameters << ',';
            parameters << "{\"name\":" << json_string(resource->getName())
                       << ",\"binding\":{\"kind\":\"descriptorTableSlot\",\"count\":1,\"index\":"
                       << known(resource->getOffset(category), "binding index") << ",\"space\":"
                       << known(resource->getBindingSpace(category), "binding space")
                       << "},\"type\":{\"kind\":\"constantBuffer\",\"size\":"
                       << known(element->getSize(), "constant buffer size") << ",\"elementType\":" << elementJson
                       << "}}";
            continue;
        }
        if (type->getKind() == slang::TypeReflection::Kind::SamplerState)
        {
            // SamplerComparisonState is a sampler kind too, but a comparison sampler binds differently (WebGPU's
            // sampler_comparison), and the document has no field for it yet: refuse it instead of calling it plain.
            if (!type->getName() || std::string(type->getName()) != "SamplerState")
                throw std::runtime_error("comparison samplers (SamplerComparisonState) are not supported");
            use.shapes = true;
            if (index) parameters << ',';
            parameters << "{\"name\":" << json_string(resource->getName())
                       << ",\"binding\":" << slot_binding_json(resource, type) << ",\"type\":{\"kind\":\"sampler\"}}";
            continue;
        }
        if (type->getKind() == slang::TypeReflection::Kind::Resource && type->getResourceShape() == SLANG_TEXTURE_2D)
        {
            // Only a plain 2D texture (no array, multisample or feedback flag). The access of a sampled Texture2D is
            // "read" (the slangc command line omits it). A sampled texture has no format in the shader, so it always
            // reports "unknown" here, whatever the API says for it (Texture2D<uint> reads back as r32ui, a
            // Texture2D<float4> as unknown); a read-write or write-only one reports the format of its artifact, which
            // is part of D17's identity: the declared [format(..)], or the one Slang infers (RWTexture2D<uint> is
            // r32ui). One that has neither (RWTexture2D<float4>) is refused.
            const char* access = nullptr;
            switch (type->getResourceAccess())
            {
            case SLANG_RESOURCE_ACCESS_READ: access = "read"; break;
            case SLANG_RESOURCE_ACCESS_READ_WRITE: access = "readWrite"; break;
            case SLANG_RESOURCE_ACCESS_WRITE: access = "write"; break;
            default: throw std::runtime_error("unsupported resource access");
            }
            const char* format = "unknown";
            if (type->getResourceAccess() != SLANG_RESOURCE_ACCESS_READ)
            {
                if (resource->getImageFormat() == SLANG_IMAGE_FORMAT_unknown)
                    throw std::runtime_error(
                        "a read-write or write-only texture needs a format, declared with [format(..)] or inferred");
                format = image_format_name(resource->getImageFormat());
            }
            use.shapes = true;
            if (index) parameters << ',';
            parameters << "{\"name\":" << json_string(resource->getName())
                       << ",\"binding\":" << slot_binding_json(resource, type)
                       << ",\"type\":{\"kind\":\"texture\",\"shape\":\"2d\",\"access\":" << json_string(access)
                       << ",\"result\":" << value_type_json(type->getResourceResultType())
                       << ",\"format\":" << json_string(format) << "}}";
            continue;
        }
        if (type->getKind() == slang::TypeReflection::Kind::Resource)
        {
            // Any other texture (1D, 3D, cube, array, multisampled, feedback, shadow, or combined with a sampler as
            // Sampler2D) is refused by name; a buffer falls through to the structured-buffer rule below.
            const auto shape = static_cast<int>(type->getResourceShape());
            const char* base = nullptr;
            switch (shape & SLANG_RESOURCE_BASE_SHAPE_MASK)
            {
            case SLANG_TEXTURE_1D: base = "1D"; break;
            case SLANG_TEXTURE_2D: base = "2D"; break;
            case SLANG_TEXTURE_3D: base = "3D"; break;
            case SLANG_TEXTURE_CUBE: base = "cube"; break;
            default: break;
            }
            if (base)
            {
                std::string text = base;
                if (shape & SLANG_TEXTURE_ARRAY_FLAG) text += " array";
                if (shape & SLANG_TEXTURE_MULTISAMPLE_FLAG) text += " multisampled";
                if (shape & SLANG_TEXTURE_FEEDBACK_FLAG) text += " feedback";
                if (shape & SLANG_TEXTURE_SHADOW_FLAG) text += " shadow";
                if (shape & SLANG_TEXTURE_COMBINED_FLAG) text += " combined with a sampler";
                throw std::runtime_error("unsupported texture shape (" + text +
                                         "); only a plain 2D texture is supported");
            }
        }
        if (type->getKind() != slang::TypeReflection::Kind::Resource ||
            type->getResourceShape() != SLANG_STRUCTURED_BUFFER ||
            type->getExplicitCounter() || resource->getCategoryCount() != 1 ||
            resource->getCategoryByIndex(0) != category || type->getSize(category) != 1)
            throw std::runtime_error("only individual structured-buffer descriptor bindings are supported");
        const char* access = nullptr;
        switch (type->getResourceAccess())
        {
        case SLANG_RESOURCE_ACCESS_READ: access = "read"; break;
        case SLANG_RESOURCE_ACCESS_READ_WRITE: access = "readWrite"; break;
        default: throw std::runtime_error("unsupported resource access");
        }
        auto element = type->getElementTypeLayout();
        const auto elementJson = type_json(element, use);
        if (index) parameters << ',';
        parameters << "{\"name\":" << json_string(resource->getName())
                   << ",\"binding\":{\"kind\":\"descriptorTableSlot\",\"count\":1,\"index\":"
                   << known(resource->getOffset(category), "binding index") << ",\"space\":"
                   << known(resource->getBindingSpace(category), "binding space")
                   << "},\"elementStride\":" << known(element->getStride(), "buffer element stride")
                   << ",\"type\":{\"kind\":\"resource\",\"baseShape\":\"structuredBuffer\",\"access\":"
                   << json_string(access) << ",\"resultType\":" << elementJson << "}}";
    }
    // Key order: schemaVersion, matrixLayout (schema 2 to 4), compiler, target, entryPoint, stage (schema 3 and 4),
    // sourceHash, artifactHash, workgroupSize (compute only), parameters, targets (fragment only).
    const int schema = use.arrays ? 4 : (use.shapes ? 3 : (use.newKinds ? 2 : 1));
    std::ostringstream output;
    output << "{\"schemaVersion\":" << schema;
    if (schema >= 2) output << ",\"matrixLayout\":" << json_string(matrix_layout_name(kMatrixLayoutMode));
    output << ",\"compiler\":" << json_string(compiler)
           << ",\"target\":" << json_string(target) << ",\"entryPoint\":" << json_string(entryName);
    if (schema >= 3) output << ",\"stage\":" << json_string(stage_name(stage));
    output << ",\"sourceHash\":\"" << fingerprint(source.data(), source.size())
           << "\",\"artifactHash\":\"" << fingerprint(code->getBufferPointer(), code->getBufferSize()) << '"';
    if (compute)
        output << ",\"workgroupSize\":[" << known(dimensions[0], "workgroup x") << ','
               << known(dimensions[1], "workgroup y") << ',' << known(dimensions[2], "workgroup z") << ']';
    output << ",\"parameters\":[" << parameters.str() << ']';
    if (!compute) output << ",\"targets\":" << targets_json(entry);
    output << '}';
    return output.str();
}

static void write_file(const char* path, const void* bytes, size_t size)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error(std::string("cannot open output: ") + path);
    output.write(static_cast<const char*>(bytes), static_cast<std::streamsize>(size));
    output.close();
    if (!output) throw std::runtime_error(std::string("cannot write output: ") + path);
}

static void describe(slang::TypeLayoutReflection* type, const std::string& path, unsigned depth = 0)
{
    if (!type || depth > 32) throw std::runtime_error("missing or excessively nested element layout");
    const auto size = known(type->getSize(), "size");
    const auto stride = known(type->getStride(), "stride");
    const auto alignment = type->getAlignment();
    if (size == 0 || stride == 0 || alignment <= 0)
        throw std::runtime_error("missing byte layout for " + path);
    std::cout << path << " size=" << size << " alignment=" << alignment << " stride=" << stride << '\n';
    for (unsigned i = 0; i < type->getFieldCount(); ++i)
    {
        auto field = type->getFieldByIndex(i);
        if (!field || !field->getName()) throw std::runtime_error("missing field");
        const auto child = path + "." + field->getName();
        std::cout << child << " offset=" << known(field->getOffset(), "field offset") << '\n';
        describe(field->getTypeLayout(), child, depth + 1);
    }
}

static void probe(slang::IGlobalSession* global, const std::string& source,
                  const char* sourcePath, SlangCompileTarget format, const char* label,
                  const char* entryName = "main", const char* artifactPath = nullptr,
                  const char* reflectionPath = nullptr, Stage stage = Stage::Compute)
{
    slang::CompilerOptionEntry option = {};
    option.name = slang::CompilerOptionName::VulkanUseEntryPointName;
    option.value.kind = slang::CompilerOptionValueKind::Int;
    option.value.intValue0 = 1;
    slang::TargetDesc target;
    target.format = format;
    if (format == SLANG_SPIRV)
    {
        target.compilerOptionEntries = &option;
        target.compilerOptionEntryCount = 1;
    }
    slang::SessionDesc desc;
    desc.targets = &target;
    desc.targetCount = 1;
    desc.defaultMatrixLayoutMode = kMatrixLayoutMode;
    ComPtr<slang::ISession> session;
    check(global->createSession(desc, session.writeRef()), "createSession failed");
    ComPtr<slang::IBlob> diag;
    auto module = session->loadModuleFromSourceString("layout_probe", sourcePath, source.c_str(), diag.writeRef());
    diagnostics(diag);
    if (!module) throw std::runtime_error("loadModuleFromSourceString failed");
    ComPtr<slang::IEntryPoint> entry;
    check(module->findEntryPointByName(entryName, entry.writeRef()), "findEntryPointByName failed");
    slang::IComponentType* components[] = {module, entry};
    ComPtr<slang::IComponentType> composite;
    auto result = session->createCompositeComponentType(components, 2, composite.writeRef(), diag.writeRef());
    check(result, "createCompositeComponentType failed", diag);
    ComPtr<slang::IComponentType> linked;
    result = composite->link(linked.writeRef(), diag.writeRef());
    check(result, "link failed", diag);
    ComPtr<slang::IBlob> code;
    result = linked->getTargetCode(0, code.writeRef(), diag.writeRef());
    check(result, "target code generation failed", diag);
    if (!code || !code->getBufferSize()) throw std::runtime_error("empty target code");
    auto layout = linked->getLayout(0, diag.writeRef());
    diagnostics(diag);
    if (!layout) throw std::runtime_error("missing program layout");
    if (artifactPath)
    {
        const auto json = reflection_json(layout, entryName, label, global->getBuildTagString(), source, code, stage);
        write_file(artifactPath, code->getBufferPointer(), code->getBufferSize());
        write_file(reflectionPath, json.data(), json.size());
        return;
    }
    if (layout->getParameterCount() == 0) throw std::runtime_error("missing fixture resources");
    std::cout << "target=" << label << " code_bytes=" << code->getBufferSize() << '\n';
    for (unsigned i = 0; i < layout->getParameterCount(); ++i)
    {
        auto resource = layout->getParameterByIndex(i);
        if (!resource || !resource->getName() || !resource->getTypeLayout())
            throw std::runtime_error("missing resource layout");
        const auto category = slang::ParameterCategory::DescriptorTableSlot;
        bool foundCategory = false;
        for (unsigned j = 0; j < resource->getCategoryCount(); ++j)
            foundCategory |= resource->getCategoryByIndex(j) == category;
        if (!foundCategory) throw std::runtime_error("missing descriptor binding category");
        std::cout << "resource=" << resource->getName()
                  << " binding=" << known(resource->getOffset(category), "binding index")
                  << " space=" << known(resource->getBindingSpace(category), "binding space") << '\n';
        describe(resource->getTypeLayout()->getElementTypeLayout(), resource->getName());
    }
}

int main(int argc, char** argv)
{
    try
    {
        if (argc != 2 && argc != 6 && argc != 7)
            throw std::runtime_error(
                "usage: gust-slang-reflect <source> <entry> <wgsl|spirv> <artifact> <json> [compute|fragment], "
                "or <layout.slang>, or --version");
        ComPtr<slang::IGlobalSession> global;
        check(slang::createGlobalSession(global.writeRef()), "createGlobalSession failed");
        if (argc == 2 && std::string(argv[1]) == "--version")
        {
            std::cout << "gust-slang-reflect schema=4 slang=" << global->getBuildTagString() << '\n';
            return 0;
        }
        std::ifstream stream(argv[1], std::ios::binary);
        if (!stream) throw std::runtime_error(std::string("cannot open source: ") + argv[1]);
        const std::string source((std::istreambuf_iterator<char>(stream)), {});
        if (stream.bad() || source.empty()) throw std::runtime_error("cannot read source or source is empty");
        if (argc == 6 || argc == 7)
        {
            const std::string target = argv[3];
            if (target != "wgsl" && target != "spirv") throw std::runtime_error("unsupported target: " + target);
            // The stage is the caller's expectation of the entry point; it is compute when the argument is absent, so
            // every existing caller is unchanged.
            Stage stage = Stage::Compute;
            if (argc == 7)
            {
                const std::string name = argv[6];
                if (name == "fragment") stage = Stage::Fragment;
                else if (name != "compute") throw std::runtime_error("unsupported stage: " + name);
            }
            probe(global, source, argv[1], target == "wgsl" ? SLANG_WGSL : SLANG_SPIRV,
                  argv[3], argv[2], argv[4], argv[5], stage);
        }
        else
        {
            std::cout << "slang=" << global->getBuildTagString() << '\n';
            probe(global, source, argv[1], SLANG_WGSL, "wgsl");
            probe(global, source, argv[1], SLANG_SPIRV, "spirv");
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Slang reflection compiler failed: " << error.what() << '\n';
        return 1;
    }
}
