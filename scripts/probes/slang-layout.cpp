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
// anywhere in the program makes the document schema 2: a top-level matrixLayout and the new type kinds.
struct SchemaUse
{
    bool newKinds = false;
};

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
        throw std::runtime_error("unsupported element type; only 32-bit scalars, vectors, matrices and structs are supported");
    }
    output << '}';
    return output.str();
}

static std::string reflection_json(slang::ProgramLayout* layout, const char* entryName,
                                   const char* target, const char* compiler,
                                   const std::string& source, slang::IBlob* code)
{
    if (layout->getEntryPointCount() != 1) throw std::runtime_error("expected exactly one entry point");
    auto entry = layout->getEntryPointByIndex(0);
    if (!entry || !entry->getName() || std::string(entry->getName()) != entryName ||
        entry->getStage() != SLANG_STAGE_COMPUTE)
        throw std::runtime_error("entry point identity or compute stage mismatch");
    for (unsigned index = 0; index < entry->getParameterCount(); ++index)
    {
        auto parameter = entry->getParameterByIndex(index);
        if (!parameter || !parameter->getSemanticName() || !parameter->getTypeLayout())
            throw std::runtime_error("only dispatch-thread-ID entry-point parameters are supported");
        std::string semantic = parameter->getSemanticName();
        for (auto& character : semantic)
            character = static_cast<char>(std::toupper(static_cast<unsigned char>(character)));
        auto type = parameter->getTypeLayout();
        if (semantic != "SV_DISPATCHTHREADID" || type->getKind() != slang::TypeReflection::Kind::Vector ||
            type->getType()->getElementCount() != 3 ||
            type->getScalarType() != slang::TypeReflection::ScalarType::UInt32)
            throw std::runtime_error("only dispatch-thread-ID entry-point parameters are supported");
        for (unsigned category = 0; category < parameter->getCategoryCount(); ++category)
            if (parameter->getCategoryByIndex(category) != slang::ParameterCategory::VaryingInput)
                throw std::runtime_error("entry-point resource/uniform bindings are unsupported");
    }
    SlangUInt dimensions[3] = {};
    entry->getComputeThreadGroupSize(3, dimensions);
    SchemaUse use;
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
    std::ostringstream output;
    output << "{\"schemaVersion\":" << (use.newKinds ? 2 : 1);
    if (use.newKinds) output << ",\"matrixLayout\":" << json_string(matrix_layout_name(kMatrixLayoutMode));
    output << ",\"compiler\":" << json_string(compiler)
           << ",\"target\":" << json_string(target) << ",\"entryPoint\":" << json_string(entryName)
           << ",\"sourceHash\":\"" << fingerprint(source.data(), source.size())
           << "\",\"artifactHash\":\"" << fingerprint(code->getBufferPointer(), code->getBufferSize())
           << "\",\"workgroupSize\":[" << known(dimensions[0], "workgroup x") << ','
           << known(dimensions[1], "workgroup y") << ',' << known(dimensions[2], "workgroup z")
           << "],\"parameters\":[" << parameters.str() << "]}";
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
                  const char* reflectionPath = nullptr)
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
        const auto json = reflection_json(layout, entryName, label, global->getBuildTagString(), source, code);
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
        if (argc != 2 && argc != 6)
            throw std::runtime_error("usage: gust-slang-reflect <source> <entry> <wgsl|spirv> <artifact> <json>, or <layout.slang>, or --version");
        ComPtr<slang::IGlobalSession> global;
        check(slang::createGlobalSession(global.writeRef()), "createGlobalSession failed");
        if (argc == 2 && std::string(argv[1]) == "--version")
        {
            std::cout << "gust-slang-reflect schema=2 slang=" << global->getBuildTagString() << '\n';
            return 0;
        }
        std::ifstream stream(argv[1], std::ios::binary);
        if (!stream) throw std::runtime_error(std::string("cannot open source: ") + argv[1]);
        const std::string source((std::istreambuf_iterator<char>(stream)), {});
        if (stream.bad() || source.empty()) throw std::runtime_error("cannot read source or source is empty");
        if (argc == 6)
        {
            const std::string target = argv[3];
            if (target != "wgsl" && target != "spirv") throw std::runtime_error("unsupported target: " + target);
            probe(global, source, argv[1], target == "wgsl" ? SLANG_WGSL : SLANG_SPIRV,
                  argv[3], argv[2], argv[4], argv[5]);
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
