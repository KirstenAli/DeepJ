#include <jni.h>
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cstring>
#include <stdexcept>

static void throwJavaRuntimeException(JNIEnv* env, const char* msg) {
    jclass exClass = env->FindClass("java/lang/RuntimeException");
    if (exClass != nullptr) {
        env->ThrowNew(exClass, msg);
    }
}

static void runMatmulF32(
        const float* a,
        const float* b,
        float* out,
        int m,
        int n,
        int k
) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            throw std::runtime_error("Metal device not available");
        }

        if (!MPSSupportsMTLDevice(device)) {
            throw std::runtime_error("Metal Performance Shaders not supported on this device");
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            throw std::runtime_error("Failed to create Metal command queue");
        }

        const NSUInteger bytesA = (NSUInteger)m * (NSUInteger)k * sizeof(float);
        const NSUInteger bytesB = (NSUInteger)k * (NSUInteger)n * sizeof(float);
        const NSUInteger bytesC = (NSUInteger)m * (NSUInteger)n * sizeof(float);

        id<MTLBuffer> bufA = [device newBufferWithBytes:a
                                                 length:bytesA
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [device newBufferWithBytes:b
                                                 length:bytesB
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [device newBufferWithLength:bytesC
                                                 options:MTLResourceStorageModeShared];

        if (bufA == nil || bufB == nil || bufC == nil) {
            throw std::runtime_error("Failed to allocate Metal buffers");
        }

        MPSMatrixDescriptor* descA =
            [MPSMatrixDescriptor matrixDescriptorWithRows:m
                                                 columns:k
                                                rowBytes:(NSUInteger)k * sizeof(float)
                                                dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor* descB =
            [MPSMatrixDescriptor matrixDescriptorWithRows:k
                                                 columns:n
                                                rowBytes:(NSUInteger)n * sizeof(float)
                                                dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor* descC =
            [MPSMatrixDescriptor matrixDescriptorWithRows:m
                                                 columns:n
                                                rowBytes:(NSUInteger)n * sizeof(float)
                                                dataType:MPSDataTypeFloat32];

        MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
        MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
        MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

        MPSMatrixMultiplication* mm =
            [[MPSMatrixMultiplication alloc] initWithDevice:device
                                             transposeLeft:NO
                                            transposeRight:NO
                                                resultRows:m
                                             resultColumns:n
                                           interiorColumns:k
                                                     alpha:1.0
                                                      beta:0.0];

        if (mm == nil) {
            throw std::runtime_error("Failed to create MPSMatrixMultiplication");
        }

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        if (commandBuffer == nil) {
            throw std::runtime_error("Failed to create Metal command buffer");
        }

        [mm encodeToCommandBuffer:commandBuffer
                       leftMatrix:matA
                      rightMatrix:matB
                     resultMatrix:matC];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (commandBuffer.status == MTLCommandBufferStatusError) {
            NSString* desc = commandBuffer.error.localizedDescription ?: @"Unknown Metal command buffer error";
            throw std::runtime_error([desc UTF8String]);
        }

        std::memcpy(out, [bufC contents], bytesC);
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_MetalNative_matmulF32(
        JNIEnv* env,
        jclass,
        jfloatArray aArr,
        jfloatArray bArr,
        jfloatArray outArr,
        jint m,
        jint n,
        jint k
) {
    if (aArr == nullptr || bArr == nullptr || outArr == nullptr) {
        throwJavaRuntimeException(env, "Null array passed to MetalNative.matmulF32");
        return;
    }

    const jsize aLen = env->GetArrayLength(aArr);
    const jsize bLen = env->GetArrayLength(bArr);
    const jsize outLen = env->GetArrayLength(outArr);

    if (aLen != m * k) {
        throwJavaRuntimeException(env, "Length mismatch for A");
        return;
    }
    if (bLen != k * n) {
        throwJavaRuntimeException(env, "Length mismatch for B");
        return;
    }
    if (outLen != m * n) {
        throwJavaRuntimeException(env, "Length mismatch for out");
        return;
    }

    jboolean aCopy = JNI_FALSE;
    jboolean bCopy = JNI_FALSE;
    jboolean outCopy = JNI_FALSE;

    jfloat* a = env->GetFloatArrayElements(aArr, &aCopy);
    jfloat* b = env->GetFloatArrayElements(bArr, &bCopy);
    jfloat* out = env->GetFloatArrayElements(outArr, &outCopy);

    if (a == nullptr || b == nullptr || out == nullptr) {
        if (a != nullptr) env->ReleaseFloatArrayElements(aArr, a, JNI_ABORT);
        if (b != nullptr) env->ReleaseFloatArrayElements(bArr, b, JNI_ABORT);
        if (out != nullptr) env->ReleaseFloatArrayElements(outArr, out, 0);
        throwJavaRuntimeException(env, "Failed to access Java float arrays");
        return;
    }

    try {
        runMatmulF32(a, b, out, m, n, k);
    } catch (const std::exception& ex) {
        env->ReleaseFloatArrayElements(aArr, a, JNI_ABORT);
        env->ReleaseFloatArrayElements(bArr, b, JNI_ABORT);
        env->ReleaseFloatArrayElements(outArr, out, 0);
        throwJavaRuntimeException(env, ex.what());
        return;
    }

    env->ReleaseFloatArrayElements(aArr, a, JNI_ABORT);
    env->ReleaseFloatArrayElements(bArr, b, JNI_ABORT);
    env->ReleaseFloatArrayElements(outArr, out, 0);
}