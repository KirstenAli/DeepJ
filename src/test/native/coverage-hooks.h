#include <cstdio>
#include <cstdlib>

static NSMutableDictionary* coveragePipelineNames() {
    static NSMutableDictionary* names = [NSMutableDictionary new];
    return names;
}

static void registerCoveragePipeline(id<MTLComputePipelineState> pipeline, NSString* name) {
    coveragePipelineNames()[[NSValue valueWithNonretainedObject:pipeline]] = name;
}

static void recordCoveragePipeline(NSString* name) {
    const char* destination = std::getenv("DEEPJ_KERNEL_COVERAGE");
    if (destination == nullptr) throw std::runtime_error("Missing Metal coverage output path");
    FILE* output = std::fopen(destination, "a");
    if (output == nullptr) throw std::runtime_error("Cannot write Metal coverage report");
    std::fprintf(output, "%s\n", [name UTF8String]);
    std::fclose(output);
}

static void selectCoveragePipeline(id<MTLComputeCommandEncoder> encoder,
                                   id<MTLComputePipelineState> pipeline) {
    [encoder setComputePipelineState:pipeline];
    NSString* name = coveragePipelineNames()[[NSValue valueWithNonretainedObject:pipeline]];
    if (name == nil) throw std::runtime_error("Coverage pipeline name is missing");
    recordCoveragePipeline(name);
}
