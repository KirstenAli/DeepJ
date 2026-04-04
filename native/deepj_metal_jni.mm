#include <jni.h>
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cstring>
#include <stdexcept>
#include <string>

// ═══════════════════════════════════════════════════════════════════════
//  Persistent Metal context (singleton)
// ═══════════════════════════════════════════════════════════════════════

struct MetalContext {
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary>      library;

    // Pipeline states for each kernel
    id<MTLComputePipelineState> addPSO;
    id<MTLComputePipelineState> subtractPSO;
    id<MTLComputePipelineState> multiplyPSO;
    id<MTLComputePipelineState> dividePSO;
    id<MTLComputePipelineState> multiplyScalarPSO;
    id<MTLComputePipelineState> sqrtPSO;
    id<MTLComputePipelineState> negPSO;
    id<MTLComputePipelineState> expPSO;
    id<MTLComputePipelineState> logPSO;
    id<MTLComputePipelineState> tanhPSO;
    id<MTLComputePipelineState> sigmoidPSO;
    id<MTLComputePipelineState> reluPSO;
    id<MTLComputePipelineState> reluBackwardPSO;
    id<MTLComputePipelineState> geluPSO;
    id<MTLComputePipelineState> geluBackwardPSO;
    id<MTLComputePipelineState> softmaxMaxPSO;
    id<MTLComputePipelineState> softmaxExpSumPSO;
    id<MTLComputePipelineState> softmaxNormPSO;
    id<MTLComputePipelineState> softmaxBackwardPSO;
    id<MTLComputePipelineState> layerNormBackwardPSO;
};

static MetalContext* gCtx = nullptr;

static NSString* metalShaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

// ── element-wise binary ──
kernel void kernel_add(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* out     [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    out[id] = a[id] + b[id];
}

kernel void kernel_subtract(device const float* a [[buffer(0)]],
                            device const float* b [[buffer(1)]],
                            device float* out     [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
    out[id] = a[id] - b[id];
}

kernel void kernel_multiply(device const float* a [[buffer(0)]],
                            device const float* b [[buffer(1)]],
                            device float* out     [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
    out[id] = a[id] * b[id];
}

kernel void kernel_divide(device const float* a [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          device float* out     [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    out[id] = a[id] / b[id];
}

// ── scalar ──
kernel void kernel_multiply_scalar(device const float* a       [[buffer(0)]],
                                   device float* out           [[buffer(1)]],
                                   device const float* scalar  [[buffer(2)]],
                                   uint id [[thread_position_in_grid]]) {
    out[id] = a[id] * scalar[0];
}

// ── unary math ──
kernel void kernel_sqrt(device const float* a [[buffer(0)]],
                        device float* out     [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = sqrt(a[id]);
}

kernel void kernel_neg(device const float* a [[buffer(0)]],
                       device float* out     [[buffer(1)]],
                       uint id [[thread_position_in_grid]]) {
    out[id] = -a[id];
}

kernel void kernel_exp(device const float* a [[buffer(0)]],
                       device float* out     [[buffer(1)]],
                       uint id [[thread_position_in_grid]]) {
    out[id] = exp(a[id]);
}

kernel void kernel_log(device const float* a [[buffer(0)]],
                       device float* out     [[buffer(1)]],
                       uint id [[thread_position_in_grid]]) {
    out[id] = log(a[id]);
}

// ── activations ──
kernel void kernel_tanh(device const float* a [[buffer(0)]],
                        device float* out     [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = tanh(a[id]);
}

kernel void kernel_sigmoid(device const float* a [[buffer(0)]],
                           device float* out     [[buffer(1)]],
                           uint id [[thread_position_in_grid]]) {
    out[id] = 1.0f / (1.0f + exp(-a[id]));
}

kernel void kernel_relu(device const float* a [[buffer(0)]],
                        device float* out     [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = max(0.0f, a[id]);
}

kernel void kernel_relu_backward(device const float* input   [[buffer(0)]],
                                 device const float* grad    [[buffer(1)]],
                                 device float* out           [[buffer(2)]],
                                 uint id [[thread_position_in_grid]]) {
    out[id] = input[id] > 0.0f ? grad[id] : 0.0f;
}

kernel void kernel_gelu(device const float* a [[buffer(0)]],
                        device float* out     [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    float x = a[id];
    float c = 0.7978845608f; // sqrt(2/pi)
    float x3 = x * x * x;
    float t = c * (x + 0.044715f * x3);
    out[id] = 0.5f * x * (1.0f + tanh(t));
}

kernel void kernel_gelu_backward(device const float* input [[buffer(0)]],
                                 device const float* grad  [[buffer(1)]],
                                 device float* out         [[buffer(2)]],
                                 uint id [[thread_position_in_grid]]) {
    float x = input[id];
    float c = 0.7978845608f;
    float x2 = x * x;
    float x3 = x2 * x;
    float t = c * (x + 0.044715f * x3);
    float tanhT = tanh(t);
    float sech2 = 1.0f - tanhT * tanhT;
    float dt_dx = c * (1.0f + 3.0f * 0.044715f * x2);
    float d_gelu = 0.5f * (1.0f + tanhT) + 0.5f * x * sech2 * dt_dx;
    out[id] = grad[id] * d_gelu;
}

// ── softmax (3-pass: max, exp+sum, normalize) ──
// Each row processed independently; dispatch rows threads for max and expsum, then n threads for norm.

kernel void kernel_softmax_max(device const float* a     [[buffer(0)]],
                               device float* rowMax      [[buffer(1)]],
                               device const uint* dims   [[buffer(2)]],
                               uint row [[thread_position_in_grid]]) {
    uint cols = dims[0];
    uint base = row * cols;
    float mx = a[base];
    for (uint c = 1; c < cols; c++) {
        mx = max(mx, a[base + c]);
    }
    rowMax[row] = mx;
}

kernel void kernel_softmax_expsum(device const float* a     [[buffer(0)]],
                                  device float* out         [[buffer(1)]],
                                  device const float* rowMax[[buffer(2)]],
                                  device float* rowSum      [[buffer(3)]],
                                  device const uint* dims   [[buffer(4)]],
                                  uint row [[thread_position_in_grid]]) {
    uint cols = dims[0];
    uint base = row * cols;
    float mx = rowMax[row];
    float s = 0.0f;
    for (uint c = 0; c < cols; c++) {
        float e = exp(a[base + c] - mx);
        out[base + c] = e;
        s += e;
    }
    rowSum[row] = s;
}

kernel void kernel_softmax_norm(device float* out          [[buffer(0)]],
                                device const float* rowSum [[buffer(1)]],
                                device const uint* dims    [[buffer(2)]],
                                uint id [[thread_position_in_grid]]) {
    uint cols = dims[0];
    uint row = id / cols;
    out[id] = out[id] / rowSum[row];
}

kernel void kernel_softmax_backward(device const float* gradOutput [[buffer(0)]],
                                    device const float* softmaxOut [[buffer(1)]],
                                    device float* out              [[buffer(2)]],
                                    device const uint* dims        [[buffer(3)]],
                                    uint row [[thread_position_in_grid]]) {
    uint cols = dims[0];
    uint base = row * cols;

    float dot = 0.0f;
    for (uint c = 0; c < cols; c++) {
        dot += gradOutput[base + c] * softmaxOut[base + c];
    }

    for (uint c = 0; c < cols; c++) {
        float s = softmaxOut[base + c];
        out[base + c] = s * (gradOutput[base + c] - dot);
    }
}

kernel void kernel_layernorm_backward(device const float* dXHat [[buffer(0)]],
                                      device const float* xHat  [[buffer(1)]],
                                      device const float* std   [[buffer(2)]],
                                      device float* out         [[buffer(3)]],
                                      device const uint* dims   [[buffer(4)]],
                                      uint row [[thread_position_in_grid]]) {
    uint cols = dims[0];
    uint base = row * cols;

    float invStd = 1.0f / std[row];
    float sumD = 0.0f;
    float sumDXHatXHat = 0.0f;

    for (uint c = 0; c < cols; c++) {
        float d = dXHat[base + c];
        sumD += d;
        sumDXHatXHat += d * xHat[base + c];
    }

    float invCols = 1.0f / (float)cols;
    for (uint c = 0; c < cols; c++) {
        float d = dXHat[base + c];
        float xh = xHat[base + c];
        out[base + c] = invStd * (d - sumD * invCols - xh * (sumDXHatXHat * invCols));
    }
}
)";

static void throwJavaRuntimeException(JNIEnv* env, const char* msg) {
    jclass exClass = env->FindClass("java/lang/RuntimeException");
    if (exClass != nullptr) {
        env->ThrowNew(exClass, msg);
    }
}

static id<MTLComputePipelineState> makePSO(id<MTLLibrary> lib, NSString* name) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (fn == nil) {
        throw std::runtime_error(std::string("Metal function not found: ") + [name UTF8String]);
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pso = [lib.device newComputePipelineStateWithFunction:fn error:&error];
    if (pso == nil) {
        NSString* desc = error.localizedDescription ?: @"Unknown error";
        throw std::runtime_error(std::string("Failed to create PSO for ") + [name UTF8String] + ": " + [desc UTF8String]);
    }
    return pso;
}

static MetalContext* getContext() {
    if (gCtx != nullptr) return gCtx;

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) throw std::runtime_error("Metal device not available");

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) throw std::runtime_error("Failed to create Metal command queue");

        NSError* error = nil;
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeFast;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            opts.fastMathEnabled = YES;
#pragma clang diagnostic pop
        }
        id<MTLLibrary> library = [device newLibraryWithSource:metalShaderSource options:opts error:&error];
        if (library == nil) {
            NSString* desc = error.localizedDescription ?: @"Unknown error";
            throw std::runtime_error(std::string("Failed to compile Metal shaders: ") + [desc UTF8String]);
        }

        gCtx = new MetalContext();
        gCtx->device  = device;
        gCtx->queue   = queue;
        gCtx->library = library;

        gCtx->addPSO            = makePSO(library, @"kernel_add");
        gCtx->subtractPSO       = makePSO(library, @"kernel_subtract");
        gCtx->multiplyPSO       = makePSO(library, @"kernel_multiply");
        gCtx->dividePSO         = makePSO(library, @"kernel_divide");
        gCtx->multiplyScalarPSO = makePSO(library, @"kernel_multiply_scalar");
        gCtx->sqrtPSO           = makePSO(library, @"kernel_sqrt");
        gCtx->negPSO            = makePSO(library, @"kernel_neg");
        gCtx->expPSO            = makePSO(library, @"kernel_exp");
        gCtx->logPSO            = makePSO(library, @"kernel_log");
        gCtx->tanhPSO           = makePSO(library, @"kernel_tanh");
        gCtx->sigmoidPSO        = makePSO(library, @"kernel_sigmoid");
        gCtx->reluPSO           = makePSO(library, @"kernel_relu");
        gCtx->reluBackwardPSO   = makePSO(library, @"kernel_relu_backward");
        gCtx->geluPSO           = makePSO(library, @"kernel_gelu");
        gCtx->geluBackwardPSO   = makePSO(library, @"kernel_gelu_backward");
        gCtx->softmaxMaxPSO     = makePSO(library, @"kernel_softmax_max");
        gCtx->softmaxExpSumPSO  = makePSO(library, @"kernel_softmax_expsum");
        gCtx->softmaxNormPSO    = makePSO(library, @"kernel_softmax_norm");
        gCtx->softmaxBackwardPSO= makePSO(library, @"kernel_softmax_backward");
        gCtx->layerNormBackwardPSO = makePSO(library, @"kernel_layernorm_backward");
    }
    return gCtx;
}

// ═══════════════════════════════════════════════════════════════════════
//  Helper: run a unary (1 input -> 1 output) compute kernel
// ═══════════════════════════════════════════════════════════════════════

static void runUnary(id<MTLComputePipelineState> pso,
                     const float* a, float* out, int n) {
    @autoreleasepool {
        MetalContext* ctx = getContext();
        NSUInteger bytes = (NSUInteger)n * sizeof(float);

        id<MTLBuffer> bufA   = [ctx->device newBufferWithBytes:a   length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufOut = [ctx->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bufA   offset:0 atIndex:0];
        [enc setBuffer:bufOut offset:0 atIndex:1];

        NSUInteger tpg = MIN((NSUInteger)n, pso.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        std::memcpy(out, [bufOut contents], bytes);
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Helper: run a binary (2 inputs -> 1 output) compute kernel
// ═══════════════════════════════════════════════════════════════════════

static void runBinary(id<MTLComputePipelineState> pso,
                      const float* a, const float* b, float* out, int n) {
    @autoreleasepool {
        MetalContext* ctx = getContext();
        NSUInteger bytes = (NSUInteger)n * sizeof(float);

        id<MTLBuffer> bufA   = [ctx->device newBufferWithBytes:a   length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB   = [ctx->device newBufferWithBytes:b   length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufOut = [ctx->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bufA   offset:0 atIndex:0];
        [enc setBuffer:bufB   offset:0 atIndex:1];
        [enc setBuffer:bufOut offset:0 atIndex:2];

        NSUInteger tpg = MIN((NSUInteger)n, pso.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        std::memcpy(out, [bufOut contents], bytes);
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Matmul (unchanged algorithm, but uses persistent context)
// ═══════════════════════════════════════════════════════════════════════

static void runMatmulF32(const float* a, const float* b, float* out,
                         int m, int n, int k) {
    @autoreleasepool {
        MetalContext* ctx = getContext();

        const NSUInteger bytesA = (NSUInteger)m * (NSUInteger)k * sizeof(float);
        const NSUInteger bytesB = (NSUInteger)k * (NSUInteger)n * sizeof(float);
        const NSUInteger bytesC = (NSUInteger)m * (NSUInteger)n * sizeof(float);

        id<MTLBuffer> bufA = [ctx->device newBufferWithBytes:a length:bytesA options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [ctx->device newBufferWithBytes:b length:bytesB options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [ctx->device newBufferWithLength:bytesC options:MTLResourceStorageModeShared];

        if (bufA == nil || bufB == nil || bufC == nil) {
            throw std::runtime_error("Failed to allocate Metal buffers");
        }

        MPSMatrixDescriptor* descA =
            [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k
                                                rowBytes:(NSUInteger)k * sizeof(float)
                                                dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* descB =
            [MPSMatrixDescriptor matrixDescriptorWithRows:k columns:n
                                                rowBytes:(NSUInteger)n * sizeof(float)
                                                dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* descC =
            [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n
                                                rowBytes:(NSUInteger)n * sizeof(float)
                                                dataType:MPSDataTypeFloat32];

        MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
        MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
        MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

        MPSMatrixMultiplication* mm =
            [[MPSMatrixMultiplication alloc] initWithDevice:ctx->device
                                             transposeLeft:NO transposeRight:NO
                                                resultRows:m resultColumns:n
                                           interiorColumns:k alpha:1.0 beta:0.0];

        id<MTLCommandBuffer> cmdBuf = [ctx->queue commandBuffer];
        [mm encodeToCommandBuffer:cmdBuf leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status == MTLCommandBufferStatusError) {
            NSString* desc = cmdBuf.error.localizedDescription ?: @"Unknown Metal command buffer error";
            throw std::runtime_error([desc UTF8String]);
        }

        std::memcpy(out, [bufC contents], bytesC);
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Softmax (3-pass GPU)
// ═══════════════════════════════════════════════════════════════════════

static void runSoftmaxRowsF32(const float* a, float* out, int rows, int cols) {
    @autoreleasepool {
        MetalContext* ctx = getContext();
        NSUInteger totalBytes = (NSUInteger)rows * (NSUInteger)cols * sizeof(float);
        NSUInteger rowBytes   = (NSUInteger)rows * sizeof(float);

        id<MTLBuffer> bufA      = [ctx->device newBufferWithBytes:a length:totalBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufOut    = [ctx->device newBufferWithLength:totalBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufMax    = [ctx->device newBufferWithLength:rowBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufSum    = [ctx->device newBufferWithLength:rowBytes options:MTLResourceStorageModeShared];
        uint32_t colsVal = (uint32_t)cols;
        id<MTLBuffer> bufDims   = [ctx->device newBufferWithBytes:&colsVal length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [ctx->queue commandBuffer];

        // Pass 1: max per row
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:ctx->softmaxMaxPSO];
            [enc setBuffer:bufA    offset:0 atIndex:0];
            [enc setBuffer:bufMax  offset:0 atIndex:1];
            [enc setBuffer:bufDims offset:0 atIndex:2];
            NSUInteger tpg = MIN((NSUInteger)rows, ctx->softmaxMaxPSO.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
        }

        // Pass 2: exp + sum per row
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:ctx->softmaxExpSumPSO];
            [enc setBuffer:bufA    offset:0 atIndex:0];
            [enc setBuffer:bufOut  offset:0 atIndex:1];
            [enc setBuffer:bufMax  offset:0 atIndex:2];
            [enc setBuffer:bufSum  offset:0 atIndex:3];
            [enc setBuffer:bufDims offset:0 atIndex:4];
            NSUInteger tpg = MIN((NSUInteger)rows, ctx->softmaxExpSumPSO.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
        }

        // Pass 3: normalize
        {
            NSUInteger total = (NSUInteger)rows * (NSUInteger)cols;
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:ctx->softmaxNormPSO];
            [enc setBuffer:bufOut  offset:0 atIndex:0];
            [enc setBuffer:bufSum  offset:0 atIndex:1];
            [enc setBuffer:bufDims offset:0 atIndex:2];
            NSUInteger tpg = MIN(total, ctx->softmaxNormPSO.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
        }

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        std::memcpy(out, [bufOut contents], totalBytes);
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Scalar multiply
// ═══════════════════════════════════════════════════════════════════════

static void runMultiplyScalarF32(const float* a, float* out, float scalar, int n) {
    @autoreleasepool {
        MetalContext* ctx = getContext();
        NSUInteger bytes = (NSUInteger)n * sizeof(float);

        id<MTLBuffer> bufA      = [ctx->device newBufferWithBytes:a length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufOut    = [ctx->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufScalar = [ctx->device newBufferWithBytes:&scalar length:sizeof(float) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->multiplyScalarPSO];
        [enc setBuffer:bufA      offset:0 atIndex:0];
        [enc setBuffer:bufOut    offset:0 atIndex:1];
        [enc setBuffer:bufScalar offset:0 atIndex:2];

        NSUInteger tpg = MIN((NSUInteger)n, ctx->multiplyScalarPSO.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        std::memcpy(out, [bufOut contents], bytes);
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  JNI helpers
// ═══════════════════════════════════════════════════════════════════════

// Two-level macro to force expansion of JNI_PREFIX before token pasting
#define JNI_PASTE_(prefix, name) prefix##name
#define JNI_PASTE(prefix, name)  JNI_PASTE_(prefix, name)
#define JNI_PREFIX Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_
#define JNI_FN(name) JNI_PASTE(JNI_PREFIX, name)

#define DEFINE_UNARY_JNI(name, pso_field) \
extern "C" JNIEXPORT void JNICALL \
JNI_FN(name)(JNIEnv* env, jclass, jfloatArray aArr, jfloatArray outArr, jint n) { \
    jfloat* a   = env->GetFloatArrayElements(aArr,   nullptr); \
    jfloat* out = env->GetFloatArrayElements(outArr,  nullptr); \
    try { \
        MetalContext* ctx = getContext(); \
        runUnary(ctx->pso_field, a, out, (int)n); \
    } catch (const std::exception& ex) { \
        env->ReleaseFloatArrayElements(aArr,   a,   JNI_ABORT); \
        env->ReleaseFloatArrayElements(outArr, out, 0); \
        throwJavaRuntimeException(env, ex.what()); \
        return; \
    } \
    env->ReleaseFloatArrayElements(aArr,   a,   JNI_ABORT); \
    env->ReleaseFloatArrayElements(outArr, out, 0); \
}

#define DEFINE_BINARY_JNI(name, pso_field) \
extern "C" JNIEXPORT void JNICALL \
JNI_FN(name)(JNIEnv* env, jclass, jfloatArray aArr, jfloatArray bArr, jfloatArray outArr, jint n) { \
    jfloat* a   = env->GetFloatArrayElements(aArr,   nullptr); \
    jfloat* b   = env->GetFloatArrayElements(bArr,   nullptr); \
    jfloat* out = env->GetFloatArrayElements(outArr,  nullptr); \
    try { \
        MetalContext* ctx = getContext(); \
        runBinary(ctx->pso_field, a, b, out, (int)n); \
    } catch (const std::exception& ex) { \
        env->ReleaseFloatArrayElements(aArr,   a,   JNI_ABORT); \
        env->ReleaseFloatArrayElements(bArr,   b,   JNI_ABORT); \
        env->ReleaseFloatArrayElements(outArr, out, 0); \
        throwJavaRuntimeException(env, ex.what()); \
        return; \
    } \
    env->ReleaseFloatArrayElements(aArr,   a,   JNI_ABORT); \
    env->ReleaseFloatArrayElements(bArr,   b,   JNI_ABORT); \
    env->ReleaseFloatArrayElements(outArr, out, 0); \
}

// ── Matmul JNI ─────────────────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_matmulF32(
        JNIEnv* env, jclass,
        jfloatArray aArr, jfloatArray bArr, jfloatArray outArr,
        jint m, jint n, jint k) {
    jfloat* a   = env->GetFloatArrayElements(aArr,   nullptr);
    jfloat* b   = env->GetFloatArrayElements(bArr,   nullptr);
    jfloat* out = env->GetFloatArrayElements(outArr,  nullptr);
    try {
        runMatmulF32(a, b, out, m, n, k);
    } catch (const std::exception& ex) {
        env->ReleaseFloatArrayElements(aArr,   a,   JNI_ABORT);
        env->ReleaseFloatArrayElements(bArr,   b,   JNI_ABORT);
        env->ReleaseFloatArrayElements(outArr, out, 0);
        throwJavaRuntimeException(env, ex.what());
        return;
    }
    env->ReleaseFloatArrayElements(aArr,   a,   JNI_ABORT);
    env->ReleaseFloatArrayElements(bArr,   b,   JNI_ABORT);
    env->ReleaseFloatArrayElements(outArr, out, 0);
}

// ── Element-wise binary JNI ────────────────────────────────────────────

DEFINE_BINARY_JNI(addF32,      addPSO)
DEFINE_BINARY_JNI(subtractF32, subtractPSO)
DEFINE_BINARY_JNI(multiplyF32, multiplyPSO)
DEFINE_BINARY_JNI(divideF32,   dividePSO)

// ── Scalar JNI ─────────────────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_multiplyScalarF32(
        JNIEnv* env, jclass,
        jfloatArray aArr, jfloatArray outArr, jfloat scalar, jint n) {
    jfloat* a   = env->GetFloatArrayElements(aArr,   nullptr);
    jfloat* out = env->GetFloatArrayElements(outArr,  nullptr);
    try {
        runMultiplyScalarF32(a, out, scalar, (int)n);
    } catch (const std::exception& ex) {
        env->ReleaseFloatArrayElements(aArr,   a,   JNI_ABORT);
        env->ReleaseFloatArrayElements(outArr, out, 0);
        throwJavaRuntimeException(env, ex.what());
        return;
    }
    env->ReleaseFloatArrayElements(aArr,   a,   JNI_ABORT);
    env->ReleaseFloatArrayElements(outArr, out, 0);
}

// ── Unary math JNI ─────────────────────────────────────────────────────

DEFINE_UNARY_JNI(sqrtF32, sqrtPSO)
DEFINE_UNARY_JNI(negF32,  negPSO)
DEFINE_UNARY_JNI(expF32,  expPSO)
DEFINE_UNARY_JNI(logF32,  logPSO)

// ── Activation JNI ─────────────────────────────────────────────────────

DEFINE_UNARY_JNI(tanhF32,    tanhPSO)
DEFINE_UNARY_JNI(sigmoidF32, sigmoidPSO)
DEFINE_UNARY_JNI(reluF32,    reluPSO)

DEFINE_BINARY_JNI(reluBackwardF32, reluBackwardPSO)
DEFINE_UNARY_JNI(geluF32,         geluPSO)
DEFINE_BINARY_JNI(geluBackwardF32, geluBackwardPSO)

// ── Softmax JNI ────────────────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_softmaxRowsF32(
        JNIEnv* env, jclass,
        jfloatArray aArr, jfloatArray outArr, jint rows, jint cols) {
    jfloat* a   = env->GetFloatArrayElements(aArr,   nullptr);
    jfloat* out = env->GetFloatArrayElements(outArr,  nullptr);
    try {
        runSoftmaxRowsF32(a, out, (int)rows, (int)cols);
    } catch (const std::exception& ex) {
        env->ReleaseFloatArrayElements(aArr,   a,   JNI_ABORT);
        env->ReleaseFloatArrayElements(outArr, out, 0);
        throwJavaRuntimeException(env, ex.what());
        return;
    }
    env->ReleaseFloatArrayElements(aArr,   a,   JNI_ABORT);
    env->ReleaseFloatArrayElements(outArr, out, 0);
}

// ═══════════════════════════════════════════════════════════════════════
//  Persistent GPU Buffer Pool + Batch Op Execution (Lazy Graph)
//
//  Buffers persist across JNI calls. Ops are batched into a single
//  MTLCommandBuffer. Data stays GPU-resident between ops.
// ═══════════════════════════════════════════════════════════════════════

#include <unordered_map>

static std::unordered_map<int, id<MTLBuffer>> gBufferPool;

// Op codes — must match ComputeGraph.java
static constexpr int OP_ADD            = 1;
static constexpr int OP_SUBTRACT       = 2;
static constexpr int OP_MULTIPLY       = 3;
static constexpr int OP_DIVIDE         = 4;
static constexpr int OP_MATMUL         = 5;
static constexpr int OP_MULTIPLY_SCALAR= 6;
static constexpr int OP_SQRT           = 7;
static constexpr int OP_NEG            = 8;
static constexpr int OP_EXP            = 9;
static constexpr int OP_LOG            = 10;
static constexpr int OP_TANH           = 11;
static constexpr int OP_SIGMOID        = 12;
static constexpr int OP_RELU           = 13;
static constexpr int OP_RELU_BACKWARD  = 14;
static constexpr int OP_GELU           = 15;
static constexpr int OP_GELU_BACKWARD  = 16;
static constexpr int OP_SOFTMAX_ROWS   = 17;
static constexpr int OP_SOFTMAX_BACKWARD = 18;
static constexpr int OP_LAYERNORM_BACKWARD = 19;

// Helper: encode a softmax 3-pass into an existing compute encoder
static void encodeSoftmaxGraph(id<MTLComputeCommandEncoder> __strong &enc,
                               id<MTLCommandBuffer> cmdBuf,
                               MetalContext* ctx,
                               id<MTLBuffer> bufIn, id<MTLBuffer> bufOut,
                               int rows, int cols) {
    NSUInteger rowBytes = (NSUInteger)rows * sizeof(float);
    id<MTLBuffer> bufMax  = [ctx->device newBufferWithLength:rowBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufSum  = [ctx->device newBufferWithLength:rowBytes options:MTLResourceStorageModeShared];
    uint32_t colsVal = (uint32_t)cols;
    id<MTLBuffer> bufDims = [ctx->device newBufferWithBytes:&colsVal length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

    if (!enc) enc = [cmdBuf computeCommandEncoder];

    // Pass 1: max per row
    [enc setComputePipelineState:ctx->softmaxMaxPSO];
    [enc setBuffer:bufIn   offset:0 atIndex:0];
    [enc setBuffer:bufMax  offset:0 atIndex:1];
    [enc setBuffer:bufDims offset:0 atIndex:2];
    NSUInteger tpg1 = MIN((NSUInteger)rows, ctx->softmaxMaxPSO.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg1, 1, 1)];

    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Pass 2: exp + sum per row
    [enc setComputePipelineState:ctx->softmaxExpSumPSO];
    [enc setBuffer:bufIn   offset:0 atIndex:0];
    [enc setBuffer:bufOut  offset:0 atIndex:1];
    [enc setBuffer:bufMax  offset:0 atIndex:2];
    [enc setBuffer:bufSum  offset:0 atIndex:3];
    [enc setBuffer:bufDims offset:0 atIndex:4];
    NSUInteger tpg2 = MIN((NSUInteger)rows, ctx->softmaxExpSumPSO.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg2, 1, 1)];

    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Pass 3: normalize
    NSUInteger total = (NSUInteger)rows * (NSUInteger)cols;
    [enc setComputePipelineState:ctx->softmaxNormPSO];
    [enc setBuffer:bufOut  offset:0 atIndex:0];
    [enc setBuffer:bufSum  offset:0 atIndex:1];
    [enc setBuffer:bufDims offset:0 atIndex:2];
    NSUInteger tpg3 = MIN(total, ctx->softmaxNormPSO.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg3, 1, 1)];
}

// ── nativeAllocBuffers ─────────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_nativeAllocBuffers(
        JNIEnv* env, jclass, jintArray idsArr, jintArray sizesArr, jint count) {
    jint* ids   = env->GetIntArrayElements(idsArr, nullptr);
    jint* sizes = env->GetIntArrayElements(sizesArr, nullptr);
    try {
        MetalContext* ctx = getContext();
        for (int i = 0; i < count; i++) {
            int bufId = ids[i];
            if (gBufferPool.find(bufId) == gBufferPool.end()) {
                NSUInteger bytes = (NSUInteger)sizes[i] * sizeof(float);
                id<MTLBuffer> buf = [ctx->device newBufferWithLength:bytes
                                     options:MTLResourceStorageModeShared];
                if (buf == nil) throw std::runtime_error("Failed to allocate Metal buffer");
                gBufferPool[bufId] = buf;
            }
        }
    } catch (const std::exception& ex) {
        env->ReleaseIntArrayElements(idsArr, ids, JNI_ABORT);
        env->ReleaseIntArrayElements(sizesArr, sizes, JNI_ABORT);
        throwJavaRuntimeException(env, ex.what());
        return;
    }
    env->ReleaseIntArrayElements(idsArr, ids, JNI_ABORT);
    env->ReleaseIntArrayElements(sizesArr, sizes, JNI_ABORT);
}

// ── nativeUploadBuffer ─────────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_nativeUploadBuffer(
        JNIEnv* env, jclass, jint bufId, jfloatArray dataArr) {
    auto it = gBufferPool.find(bufId);
    if (it == gBufferPool.end()) {
        throwJavaRuntimeException(env, "Buffer not found for upload");
        return;
    }
    jint len = env->GetArrayLength(dataArr);
    jfloat* data = env->GetFloatArrayElements(dataArr, nullptr);
    std::memcpy([it->second contents], data, (size_t)len * sizeof(float));
    env->ReleaseFloatArrayElements(dataArr, data, JNI_ABORT);
}

// ── nativeDownloadBuffer ───────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_nativeDownloadBuffer(
        JNIEnv* env, jclass, jint bufId, jfloatArray outArr) {
    auto it = gBufferPool.find(bufId);
    if (it == gBufferPool.end()) {
        throwJavaRuntimeException(env, "Buffer not found for download");
        return;
    }
    jint len = env->GetArrayLength(outArr);
    jfloat* out = env->GetFloatArrayElements(outArr, nullptr);
    std::memcpy(out, [it->second contents], (size_t)len * sizeof(float));
    env->ReleaseFloatArrayElements(outArr, out, 0);
}

// ── nativeReleaseBuffers ───────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_nativeReleaseBuffers(
        JNIEnv* env, jclass, jintArray idsArr, jint count) {
    jint* ids = env->GetIntArrayElements(idsArr, nullptr);
    for (int i = 0; i < count; i++) {
        gBufferPool.erase(ids[i]);
    }
    env->ReleaseIntArrayElements(idsArr, ids, JNI_ABORT);
}


// ── nativeFlushOps: batch execute all ops in one MTLCommandBuffer ──────

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_nativeFlushOps(
        JNIEnv* env, jclass, jintArray cmdStreamArr, jint cmdStreamLength) {
    jint* cmd = env->GetIntArrayElements(cmdStreamArr, nullptr);
    try {
        @autoreleasepool {
            MetalContext* ctx = getContext();
            id<MTLCommandBuffer> cmdBuf = [ctx->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = nil;

            int pos = 0;
            while (pos < cmdStreamLength) {
                int opCode = cmd[pos];

                switch (opCode) {

                // ── Binary element-wise: [op, a, b, out, n] ──────────
                case OP_ADD: case OP_SUBTRACT: case OP_MULTIPLY: case OP_DIVIDE:
                case OP_RELU_BACKWARD: case OP_GELU_BACKWARD: {
                    int aId = cmd[pos+1], bId = cmd[pos+2], outId = cmd[pos+3];
                    int n = cmd[pos+4];

                    id<MTLComputePipelineState> pso;
                    switch (opCode) {
                        case OP_ADD:            pso = ctx->addPSO; break;
                        case OP_SUBTRACT:       pso = ctx->subtractPSO; break;
                        case OP_MULTIPLY:       pso = ctx->multiplyPSO; break;
                        case OP_DIVIDE:         pso = ctx->dividePSO; break;
                        case OP_RELU_BACKWARD:  pso = ctx->reluBackwardPSO; break;
                        case OP_GELU_BACKWARD:  pso = ctx->geluBackwardPSO; break;
                        default: pso = ctx->addPSO; break;
                    }

                    if (!enc) enc = [cmdBuf computeCommandEncoder];
                    [enc setComputePipelineState:pso];
                    [enc setBuffer:gBufferPool[aId]   offset:0 atIndex:0];
                    [enc setBuffer:gBufferPool[bId]   offset:0 atIndex:1];
                    [enc setBuffer:gBufferPool[outId] offset:0 atIndex:2];
                    NSUInteger tpg = MIN((NSUInteger)n, pso.maxTotalThreadsPerThreadgroup);
                    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
                    pos += 5;
                    break;
                }

                // ── Unary: [op, in, out, n] ──────────────────────────
                case OP_SQRT: case OP_NEG: case OP_EXP: case OP_LOG:
                case OP_TANH: case OP_SIGMOID: case OP_RELU: case OP_GELU: {
                    int inId = cmd[pos+1], outId = cmd[pos+2];
                    int n = cmd[pos+3];

                    id<MTLComputePipelineState> pso;
                    switch (opCode) {
                        case OP_SQRT:    pso = ctx->sqrtPSO; break;
                        case OP_NEG:     pso = ctx->negPSO; break;
                        case OP_EXP:     pso = ctx->expPSO; break;
                        case OP_LOG:     pso = ctx->logPSO; break;
                        case OP_TANH:    pso = ctx->tanhPSO; break;
                        case OP_SIGMOID: pso = ctx->sigmoidPSO; break;
                        case OP_RELU:    pso = ctx->reluPSO; break;
                        case OP_GELU:    pso = ctx->geluPSO; break;
                        default: pso = ctx->sqrtPSO; break;
                    }

                    if (!enc) enc = [cmdBuf computeCommandEncoder];
                    [enc setComputePipelineState:pso];
                    [enc setBuffer:gBufferPool[inId]  offset:0 atIndex:0];
                    [enc setBuffer:gBufferPool[outId] offset:0 atIndex:1];
                    NSUInteger tpg = MIN((NSUInteger)n, pso.maxTotalThreadsPerThreadgroup);
                    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
                    pos += 4;
                    break;
                }

                // ── Scalar multiply: [op, in, out, scalarBits, n] ────
                case OP_MULTIPLY_SCALAR: {
                    int inId = cmd[pos+1], outId = cmd[pos+2];
                    float scalar;
                    int bits = cmd[pos+3];
                    std::memcpy(&scalar, &bits, sizeof(float));
                    int n = cmd[pos+4];

                    id<MTLBuffer> scalarBuf = [ctx->device newBufferWithBytes:&scalar
                                               length:sizeof(float)
                                               options:MTLResourceStorageModeShared];

                    if (!enc) enc = [cmdBuf computeCommandEncoder];
                    [enc setComputePipelineState:ctx->multiplyScalarPSO];
                    [enc setBuffer:gBufferPool[inId]  offset:0 atIndex:0];
                    [enc setBuffer:gBufferPool[outId] offset:0 atIndex:1];
                    [enc setBuffer:scalarBuf          offset:0 atIndex:2];
                    NSUInteger tpg = MIN((NSUInteger)n, ctx->multiplyScalarPSO.maxTotalThreadsPerThreadgroup);
                    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
                    pos += 5;
                    break;
                }

                // ── Matmul: [op, a, b, out, m, n, k] ────────────────
                case OP_MATMUL: {
                    // MPS needs its own encoding — end compute encoder first
                    if (enc) { [enc endEncoding]; enc = nil; }

                    int aId = cmd[pos+1], bId = cmd[pos+2], outId = cmd[pos+3];
                    int m = cmd[pos+4], n = cmd[pos+5], k = cmd[pos+6];

                    MPSMatrixDescriptor* descA =
                        [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k
                            rowBytes:(NSUInteger)k * sizeof(float) dataType:MPSDataTypeFloat32];
                    MPSMatrixDescriptor* descB =
                        [MPSMatrixDescriptor matrixDescriptorWithRows:k columns:n
                            rowBytes:(NSUInteger)n * sizeof(float) dataType:MPSDataTypeFloat32];
                    MPSMatrixDescriptor* descC =
                        [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n
                            rowBytes:(NSUInteger)n * sizeof(float) dataType:MPSDataTypeFloat32];

                    MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:gBufferPool[aId] descriptor:descA];
                    MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:gBufferPool[bId] descriptor:descB];
                    MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:gBufferPool[outId] descriptor:descC];

                    MPSMatrixMultiplication* mm =
                        [[MPSMatrixMultiplication alloc] initWithDevice:ctx->device
                            transposeLeft:NO transposeRight:NO
                            resultRows:m resultColumns:n interiorColumns:k
                            alpha:1.0 beta:0.0];

                    [mm encodeToCommandBuffer:cmdBuf leftMatrix:matA rightMatrix:matB resultMatrix:matC];
                    pos += 7;
                    break;
                }

                // ── Softmax rows: [op, in, out, rows, cols] ──────────
                case OP_SOFTMAX_ROWS: {
                    int inId = cmd[pos+1], outId = cmd[pos+2];
                    int rows = cmd[pos+3], cols = cmd[pos+4];

                    encodeSoftmaxGraph(enc, cmdBuf, ctx,
                                       gBufferPool[inId], gBufferPool[outId],
                                       rows, cols);
                    pos += 5;
                    break;
                }

                // ── Softmax backward: [op, grad, softmax, out, rows, cols] ──
                case OP_SOFTMAX_BACKWARD: {
                    int gradId = cmd[pos+1], softmaxId = cmd[pos+2], outId = cmd[pos+3];
                    int rows = cmd[pos+4], cols = cmd[pos+5];
                    uint32_t colsVal = (uint32_t)cols;
                    id<MTLBuffer> dimsBuf = [ctx->device newBufferWithBytes:&colsVal
                                             length:sizeof(uint32_t)
                                             options:MTLResourceStorageModeShared];

                    if (!enc) enc = [cmdBuf computeCommandEncoder];
                    [enc setComputePipelineState:ctx->softmaxBackwardPSO];
                    [enc setBuffer:gBufferPool[gradId]    offset:0 atIndex:0];
                    [enc setBuffer:gBufferPool[softmaxId] offset:0 atIndex:1];
                    [enc setBuffer:gBufferPool[outId]     offset:0 atIndex:2];
                    [enc setBuffer:dimsBuf                offset:0 atIndex:3];
                    NSUInteger tpg = MIN((NSUInteger)rows, ctx->softmaxBackwardPSO.maxTotalThreadsPerThreadgroup);
                    [enc dispatchThreads:MTLSizeMake(rows, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
                    pos += 6;
                    break;
                }

                // ── LayerNorm backward: [op, dXHat, xHat, std, out, rows, cols] ──
                case OP_LAYERNORM_BACKWARD: {
                    int dXHatId = cmd[pos+1], xHatId = cmd[pos+2], stdId = cmd[pos+3], outId = cmd[pos+4];
                    int rows = cmd[pos+5], cols = cmd[pos+6];
                    uint32_t colsVal = (uint32_t)cols;
                    id<MTLBuffer> dimsBuf = [ctx->device newBufferWithBytes:&colsVal
                                             length:sizeof(uint32_t)
                                             options:MTLResourceStorageModeShared];

                    if (!enc) enc = [cmdBuf computeCommandEncoder];
                    [enc setComputePipelineState:ctx->layerNormBackwardPSO];
                    [enc setBuffer:gBufferPool[dXHatId] offset:0 atIndex:0];
                    [enc setBuffer:gBufferPool[xHatId]  offset:0 atIndex:1];
                    [enc setBuffer:gBufferPool[stdId]   offset:0 atIndex:2];
                    [enc setBuffer:gBufferPool[outId]   offset:0 atIndex:3];
                    [enc setBuffer:dimsBuf              offset:0 atIndex:4];
                    NSUInteger tpg = MIN((NSUInteger)rows, ctx->layerNormBackwardPSO.maxTotalThreadsPerThreadgroup);
                    [enc dispatchThreads:MTLSizeMake(rows, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
                    pos += 7;
                    break;
                }

                default:
                    throw std::runtime_error("Unknown op code in graph: " + std::to_string(opCode));
                }
            }

            if (enc) [enc endEncoding];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];

            if (cmdBuf.status == MTLCommandBufferStatusError) {
                NSString* desc = cmdBuf.error.localizedDescription ?: @"Unknown error";
                throw std::runtime_error(std::string("Metal command buffer error: ") + [desc UTF8String]);
            }
        }
    } catch (const std::exception& ex) {
        env->ReleaseIntArrayElements(cmdStreamArr, cmd, JNI_ABORT);
        throwJavaRuntimeException(env, ex.what());
        return;
    }
    env->ReleaseIntArrayElements(cmdStreamArr, cmd, JNI_ABORT);
}
