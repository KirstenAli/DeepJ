#include <jni.h>
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

struct MetalContext {
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary>      library;

    id<MTLComputePipelineState> addPSO;
    id<MTLComputePipelineState> subtractPSO;
    id<MTLComputePipelineState> multiplyPSO;
    id<MTLComputePipelineState> dividePSO;
    id<MTLComputePipelineState> multiplyScalarPSO;
    id<MTLComputePipelineState> addScalarPSO;
    id<MTLComputePipelineState> divideScalarPSO;
    id<MTLComputePipelineState> transposePSO;
    id<MTLComputePipelineState> addRowVectorPSO;
    id<MTLComputePipelineState> addBroadcastColsPSO;
    id<MTLComputePipelineState> subtractBroadcastColsPSO;
    id<MTLComputePipelineState> divideBroadcastColsPSO;
    id<MTLComputePipelineState> multiplyBroadcastColsPSO;
    id<MTLComputePipelineState> multiplyBroadcastRowsPSO;
    id<MTLComputePipelineState> sumRowsPSO;
    id<MTLComputePipelineState> sumAlongRowsPSO;
    id<MTLComputePipelineState> meanAlongRowsPSO;
    id<MTLComputePipelineState> varianceAlongRowsPSO;
    id<MTLComputePipelineState> maxAlongRowsPSO;
    id<MTLComputePipelineState> sumAbsPSO;
    id<MTLComputePipelineState> crossEntropyLossPSO;
    id<MTLComputePipelineState> crossEntropyGradPSO;
    id<MTLComputePipelineState> clampPSO;
    id<MTLComputePipelineState> powPSO;
    id<MTLComputePipelineState> scatterAddRowsPSO;
    id<MTLComputePipelineState> scatterAddRowsAtomicPSO;
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
    id<MTLComputePipelineState> adamWUpdatePSO;
};

static MetalContext* gCtx = nullptr;
static std::mutex gContextMutex;

static NSString* metalShaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

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

kernel void kernel_multiply_scalar(device const float* a       [[buffer(0)]],
                                   device float* out           [[buffer(1)]],
                                   device const float* scalar  [[buffer(2)]],
                                   uint id [[thread_position_in_grid]]) {
    out[id] = a[id] * scalar[0];
}

kernel void kernel_add_scalar(device const float* a       [[buffer(0)]],
                              device float* out           [[buffer(1)]],
                              device const float* scalar  [[buffer(2)]],
                              uint id [[thread_position_in_grid]]) {
    out[id] = a[id] + scalar[0];
}

kernel void kernel_divide_scalar(device const float* a       [[buffer(0)]],
                                 device float* out           [[buffer(1)]],
                                 device const float* scalar  [[buffer(2)]],
                                 uint id [[thread_position_in_grid]]) {
    out[id] = a[id] / scalar[0];
}

kernel void kernel_transpose(device const float* a      [[buffer(0)]],
                             device float* out          [[buffer(1)]],
                             device const uint2* dims   [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {
    uint rows = dims[0].x;
    uint cols = dims[0].y;
    uint total = rows * cols;
    if (id >= total) return;

    uint r = id / cols;
    uint c = id % cols;
    out[c * rows + r] = a[id];
}

kernel void kernel_add_row_vector(device const float* a      [[buffer(0)]],
                                  device const float* rowVec [[buffer(1)]],
                                  device float* out          [[buffer(2)]],
                                  device const uint2* dims   [[buffer(3)]],
                                  uint id [[thread_position_in_grid]]) {
    uint rows = dims[0].x;
    uint cols = dims[0].y;
    uint total = rows * cols;
    if (id >= total) return;

    uint c = id % cols;
    out[id] = a[id] + rowVec[c];
}

kernel void kernel_add_broadcast_cols(device const float* a      [[buffer(0)]],
                                      device const float* colVec [[buffer(1)]],
                                      device float* out          [[buffer(2)]],
                                      device const uint2* dims   [[buffer(3)]],
                                      uint id [[thread_position_in_grid]]) {
    uint rows = dims[0].x;
    uint cols = dims[0].y;
    uint total = rows * cols;
    if (id >= total) return;

    uint r = id / cols;
    out[id] = a[id] + colVec[r];
}

kernel void kernel_subtract_broadcast_cols(device const float* a      [[buffer(0)]],
                                           device const float* colVec [[buffer(1)]],
                                           device float* out          [[buffer(2)]],
                                           device const uint2* dims   [[buffer(3)]],
                                           uint id [[thread_position_in_grid]]) {
    uint rows = dims[0].x;
    uint cols = dims[0].y;
    uint total = rows * cols;
    if (id >= total) return;

    uint r = id / cols;
    out[id] = a[id] - colVec[r];
}

kernel void kernel_divide_broadcast_cols(device const float* a      [[buffer(0)]],
                                         device const float* colVec [[buffer(1)]],
                                         device float* out          [[buffer(2)]],
                                         device const uint2* dims   [[buffer(3)]],
                                         uint id [[thread_position_in_grid]]) {
    uint rows = dims[0].x;
    uint cols = dims[0].y;
    uint total = rows * cols;
    if (id >= total) return;

    uint r = id / cols;
    out[id] = a[id] / colVec[r];
}

kernel void kernel_multiply_broadcast_cols(device const float* a      [[buffer(0)]],
                                           device const float* colVec [[buffer(1)]],
                                           device float* out          [[buffer(2)]],
                                           device const uint2* dims   [[buffer(3)]],
                                           uint id [[thread_position_in_grid]]) {
    uint rows = dims[0].x;
    uint cols = dims[0].y;
    uint total = rows * cols;
    if (id >= total) return;

    uint r = id / cols;
    out[id] = a[id] * colVec[r];
}

kernel void kernel_multiply_broadcast_rows(device const float* a      [[buffer(0)]],
                                           device const float* rowVec [[buffer(1)]],
                                           device float* out          [[buffer(2)]],
                                           device const uint2* dims   [[buffer(3)]],
                                           uint id [[thread_position_in_grid]]) {
    uint rows = dims[0].x;
    uint cols = dims[0].y;
    uint total = rows * cols;
    if (id >= total) return;

    uint c = id % cols;
    out[id] = a[id] * rowVec[c];
}

kernel void kernel_sum_rows(device const float* a      [[buffer(0)]],
                            device float* out          [[buffer(1)]],
                            device const uint2* dims   [[buffer(2)]],
                            uint3 gid [[thread_position_in_grid]],
                            uint3 tid3 [[thread_position_in_threadgroup]],
                            uint3 tptg [[threads_per_threadgroup]]) {
    uint rows = dims[0].x;
    uint cols = dims[0].y;
    uint col = gid.x;
    uint tid = tid3.y;
    if (col >= cols) return;

    threadgroup float scratch[1024];

    float local = 0.0f;
    for (uint r = tid; r < rows; r += tptg.y) {
        local += a[r * cols + col];
    }
    scratch[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tptg.y >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) out[col] = scratch[0];
}

kernel void kernel_mean_along_rows(device const float* a      [[buffer(0)]],
                                   device float* out          [[buffer(1)]],
                                   device const uint2* dims   [[buffer(2)]],
                                   uint3 gid [[thread_position_in_grid]],
                                   uint3 tid3 [[thread_position_in_threadgroup]],
                                   uint3 tptg [[threads_per_threadgroup]]) {
    uint rows = dims[0].x;
    uint cols = dims[0].y;
    uint row = gid.y;
    uint tid = tid3.x;
    if (row >= rows) return;

    threadgroup float scratch[1024];

    uint base = row * cols;
    float local = 0.0f;
    for (uint c = tid; c < cols; c += tptg.x) {
        local += a[base + c];
    }
    scratch[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tptg.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) out[row] = scratch[0] / (float)cols;
}

kernel void kernel_sum_along_rows(device const float* a      [[buffer(0)]],
                                  device float* out          [[buffer(1)]],
                                  device const uint2* dims   [[buffer(2)]],
                                  uint3 gid [[thread_position_in_grid]],
                                  uint3 tid3 [[thread_position_in_threadgroup]],
                                  uint3 tptg [[threads_per_threadgroup]]) {
    uint rows = dims[0].x;
    uint cols = dims[0].y;
    uint row = gid.y;
    uint tid = tid3.x;
    if (row >= rows) return;

    threadgroup float scratch[1024];

    uint base = row * cols;
    float local = 0.0f;
    for (uint c = tid; c < cols; c += tptg.x) {
        local += a[base + c];
    }
    scratch[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tptg.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) out[row] = scratch[0];
}

kernel void kernel_variance_along_rows(device const float* a      [[buffer(0)]],
                                       device float* out          [[buffer(1)]],
                                       device const uint2* dims   [[buffer(2)]],
                                       uint3 gid [[thread_position_in_grid]],
                                       uint3 tid3 [[thread_position_in_threadgroup]],
                                       uint3 tptg [[threads_per_threadgroup]]) {
    uint rows = dims[0].x;
    uint cols = dims[0].y;
    uint row = gid.y;
    uint tid = tid3.x;
    if (row >= rows) return;

    threadgroup float scratch[1024];

    uint base = row * cols;
    float localSum = 0.0f;
    for (uint c = tid; c < cols; c += tptg.x) {
        localSum += a[base + c];
    }
    scratch[tid] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tptg.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = scratch[0] / (float)cols;

    float localVar = 0.0f;
    for (uint c = tid; c < cols; c += tptg.x) {
        float d = a[base + c] - mean;
        localVar += d * d;
    }
    scratch[tid] = localVar;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tptg.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) out[row] = scratch[0] / (float)cols;
}

kernel void kernel_max_along_rows(device const float* a      [[buffer(0)]],
                                  device float* out          [[buffer(1)]],
                                  device const uint2* dims   [[buffer(2)]],
                                  uint3 gid [[thread_position_in_grid]],
                                  uint3 tid3 [[thread_position_in_threadgroup]],
                                  uint3 tptg [[threads_per_threadgroup]]) {
    uint rows = dims[0].x;
    uint cols = dims[0].y;
    uint row = gid.y;
    uint tid = tid3.x;
    if (row >= rows) return;

    threadgroup float scratch[1024];

    uint base = row * cols;
    float localMax = -INFINITY;
    for (uint c = tid; c < cols; c += tptg.x) {
        localMax = max(localMax, a[base + c]);
    }
    scratch[tid] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tptg.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) out[row] = scratch[0];
}

kernel void kernel_sum_abs(device const float* a      [[buffer(0)]],
                           device float* out          [[buffer(1)]],
                           device const uint2* dims   [[buffer(2)]],
                           uint3 gid [[thread_position_in_grid]],
                           uint3 tid3 [[thread_position_in_threadgroup]],
                           uint3 tptg [[threads_per_threadgroup]]) {
    uint rows = dims[0].x;
    uint cols = dims[0].y;
    uint row = gid.y;
    uint tid = tid3.x;
    if (row >= rows) return;

    threadgroup float scratch[1024];

    uint base = row * cols;
    float local = 0.0f;
    for (uint c = tid; c < cols; c += tptg.x) {
        local += fabs(a[base + c]);
    }
    scratch[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tptg.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) out[row] = scratch[0];
}

kernel void kernel_cross_entropy_loss(device const float* logits [[buffer(0)]],
                                      device const float* targets [[buffer(1)]],
                                      device float* out           [[buffer(2)]],
                                      device const uint2* dims    [[buffer(3)]],
                                      uint3 gid [[thread_position_in_grid]],
                                      uint3 tid3 [[thread_position_in_threadgroup]],
                                      uint3 tptg [[threads_per_threadgroup]]) {
    uint rows = dims[0].x;
    uint cols = dims[0].y;
    uint row = gid.y;
    uint tid = tid3.x;
    if (row >= rows) return;

    threadgroup float scratch[1024];

    uint base = row * cols;
    float localMax = -INFINITY;
    for (uint c = tid; c < cols; c += tptg.x) {
        localMax = max(localMax, logits[base + c]);
    }
    scratch[tid] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tptg.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maxVal = scratch[0];

    float localSumExp = 0.0f;
    for (uint c = tid; c < cols; c += tptg.x) {
        localSumExp += exp(logits[base + c] - maxVal);
    }
    scratch[tid] = localSumExp;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tptg.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sumExp = scratch[0];

    if (tid == 0) {
        int target = (int)targets[row];
        if (target < 0 || target >= (int)cols) {
            out[row] = NAN;
            return;
        }
        out[row] = log(sumExp) + maxVal - logits[base + (uint)target];
    }
}

kernel void kernel_cross_entropy_gradient(device const float* logits [[buffer(0)]],
                                          device const float* targets [[buffer(1)]],
                                          device float* out           [[buffer(2)]],
                                          device const uint2* dims    [[buffer(3)]],
                                          uint3 gid [[thread_position_in_grid]],
                                          uint3 tid3 [[thread_position_in_threadgroup]],
                                          uint3 tptg [[threads_per_threadgroup]]) {
    uint rows = dims[0].x;
    uint cols = dims[0].y;
    uint row = gid.y;
    uint tid = tid3.x;
    if (row >= rows) return;

    threadgroup float scratch[1024];

    uint base = row * cols;
    int target = (int)targets[row];
    if (target < 0 || target >= (int)cols) {
        for (uint c = tid; c < cols; c += tptg.x) {
            out[base + c] = NAN;
        }
        return;
    }

    float localMax = -INFINITY;
    for (uint c = tid; c < cols; c += tptg.x) {
        localMax = max(localMax, logits[base + c]);
    }
    scratch[tid] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tptg.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maxVal = scratch[0];

    float localSumExp = 0.0f;
    for (uint c = tid; c < cols; c += tptg.x) {
        localSumExp += exp(logits[base + c] - maxVal);
    }
    scratch[tid] = localSumExp;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tptg.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sumExp = scratch[0];

    float invRows = 1.0f / (float)rows;
    for (uint c = tid; c < cols; c += tptg.x) {
        float p = exp(logits[base + c] - maxVal) / sumExp;
        if ((int)c == target) p -= 1.0f;
        out[base + c] = p * invRows;
    }
}

kernel void kernel_clamp(device const float* a        [[buffer(0)]],
                         device float* out            [[buffer(1)]],
                         device const float2* minMax  [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    float lo = minMax[0].x;
    float hi = minMax[0].y;
    out[id] = min(hi, max(lo, a[id]));
}

kernel void kernel_pow(device const float* a        [[buffer(0)]],
                       device float* out            [[buffer(1)]],
                       device const float* exponent [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    out[id] = pow(a[id], exponent[0]);
}

kernel void kernel_scatter_add_rows(device float* target      [[buffer(0)]],
                                    device const float* indices [[buffer(1)]],
                                    device const float* grad    [[buffer(2)]],
                                    device const uint3* dims    [[buffer(3)]],
                                    uint id [[thread_position_in_grid]]) {
    uint targetRows = dims[0].x;
    uint targetCols = dims[0].y;
    uint nIdx = dims[0].z;

    uint total = nIdx * targetCols;
    if (id >= total) return;

    uint i = id / targetCols;
    uint c = id % targetCols;
    uint row = (uint)indices[i];
    if (row >= targetRows) return;

    target[row * targetCols + c] += grad[i * targetCols + c];
}

inline void atomic_add_f32(device atomic_uint* addr, float value) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    while (true) {
        float current = as_type<float>(expected);
        uint desired = as_type<uint>(current + value);
        if (atomic_compare_exchange_weak_explicit(
                addr, &expected, desired,
                memory_order_relaxed, memory_order_relaxed)) {
            return;
        }
    }
}

kernel void kernel_scatter_add_rows_atomic(device float* target        [[buffer(0)]],
                                           device const float* indices [[buffer(1)]],
                                           device const float* grad    [[buffer(2)]],
                                           device const uint3* dims    [[buffer(3)]],
                                           uint id [[thread_position_in_grid]]) {
    uint targetRows = dims[0].x;
    uint targetCols = dims[0].y;
    uint nIdx = dims[0].z;

    uint total = nIdx * targetCols;
    if (id >= total) return;

    uint i = id / targetCols;
    uint c = id % targetCols;
    uint row = (uint)indices[i];
    if (row >= targetRows) return;

    uint flat = row * targetCols + c;
    device atomic_uint* targetAtomic = (device atomic_uint*)target;
    atomic_add_f32(&targetAtomic[flat], grad[i * targetCols + c]);
}

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
    float sqrtTwoOverPi = 0.7978845608f;
    float x3 = x * x * x;
    float t = sqrtTwoOverPi * (x + 0.044715f * x3);
    out[id] = 0.5f * x * (1.0f + tanh(t));
}

kernel void kernel_gelu_backward(device const float* input [[buffer(0)]],
                                 device const float* grad  [[buffer(1)]],
                                 device float* out         [[buffer(2)]],
                                 uint id [[thread_position_in_grid]]) {
    float x = input[id];
    float sqrtTwoOverPi = 0.7978845608f;
    float x2 = x * x;
    float x3 = x2 * x;
    float t = sqrtTwoOverPi * (x + 0.044715f * x3);
    float tanhT = tanh(t);
    float sech2 = 1.0f - tanhT * tanhT;
    float dt_dx = sqrtTwoOverPi * (1.0f + 3.0f * 0.044715f * x2);
    float d_gelu = 0.5f * (1.0f + tanhT) + 0.5f * x * sech2 * dt_dx;
    out[id] = grad[id] * d_gelu;
}

kernel void kernel_softmax_max(device const float* a     [[buffer(0)]],
                               device float* rowMax      [[buffer(1)]],
                               device const uint* dims   [[buffer(2)]],
                               uint3 gid [[thread_position_in_grid]],
                               uint3 tid3 [[thread_position_in_threadgroup]],
                               uint3 tptg [[threads_per_threadgroup]]) {
    uint cols = dims[0];
    uint row = gid.y;
    uint tid = tid3.x;

    threadgroup float scratch[1024];

    uint base = row * cols;
    float localMax = -INFINITY;
    for (uint c = tid; c < cols; c += tptg.x) {
        localMax = max(localMax, a[base + c]);
    }
    scratch[tid] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tptg.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) rowMax[row] = scratch[0];
}

kernel void kernel_softmax_expsum(device const float* a     [[buffer(0)]],
                                  device float* out         [[buffer(1)]],
                                  device const float* rowMax[[buffer(2)]],
                                  device float* rowSum      [[buffer(3)]],
                                  device const uint* dims   [[buffer(4)]],
                                  uint3 gid [[thread_position_in_grid]],
                                  uint3 tid3 [[thread_position_in_threadgroup]],
                                  uint3 tptg [[threads_per_threadgroup]]) {
    uint cols = dims[0];
    uint row = gid.y;
    uint tid = tid3.x;

    threadgroup float scratch[1024];

    uint base = row * cols;
    float mx = rowMax[row];
    float local = 0.0f;
    for (uint c = tid; c < cols; c += tptg.x) {
        float e = exp(a[base + c] - mx);
        out[base + c] = e;
        local += e;
    }
    scratch[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tptg.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) rowSum[row] = scratch[0];
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
                                    uint3 gid [[thread_position_in_grid]],
                                    uint3 tid3 [[thread_position_in_threadgroup]],
                                    uint3 tptg [[threads_per_threadgroup]]) {
    uint cols = dims[0];
    uint row = gid.y;
    uint tid = tid3.x;

    threadgroup float scratch[1024];

    uint base = row * cols;

    float localDot = 0.0f;
    for (uint c = tid; c < cols; c += tptg.x) {
        localDot += gradOutput[base + c] * softmaxOut[base + c];
    }
    scratch[tid] = localDot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tptg.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float dot = scratch[0];

    for (uint c = tid; c < cols; c += tptg.x) {
        float s = softmaxOut[base + c];
        out[base + c] = s * (gradOutput[base + c] - dot);
    }
}

kernel void kernel_layernorm_backward(device const float* dXHat [[buffer(0)]],
                                      device const float* xHat  [[buffer(1)]],
                                      device const float* std   [[buffer(2)]],
                                      device float* out         [[buffer(3)]],
                                      device const uint* dims   [[buffer(4)]],
                                      uint3 gid [[thread_position_in_grid]],
                                      uint3 tid3 [[thread_position_in_threadgroup]],
                                      uint3 tptg [[threads_per_threadgroup]]) {
    uint cols = dims[0];
    uint row = gid.y;
    uint tid = tid3.x;

    threadgroup float scratchA[1024];
    threadgroup float scratchB[1024];

    uint base = row * cols;

    float invStd = 1.0f / std[row];
    float localSumD = 0.0f;
    float localSumDXHatXHat = 0.0f;

    for (uint c = tid; c < cols; c += tptg.x) {
        float d = dXHat[base + c];
        localSumD += d;
        localSumDXHatXHat += d * xHat[base + c];
    }
    scratchA[tid] = localSumD;
    scratchB[tid] = localSumDXHatXHat;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tptg.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratchA[tid] += scratchA[tid + stride];
            scratchB[tid] += scratchB[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sumD = scratchA[0];
    float sumDXHatXHat = scratchB[0];

    float invCols = 1.0f / (float)cols;
    for (uint c = tid; c < cols; c += tptg.x) {
        float d = dXHat[base + c];
        float xh = xHat[base + c];
        out[base + c] = invStd * (d - sumD * invCols - xh * (sumDXHatXHat * invCols));
    }
}

struct AdamWParams {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weightDecay;
    float bc1;
    float bc2;
};

kernel void kernel_adamw_update(device float* w            [[buffer(0)]],
                                device const float* g      [[buffer(1)]],
                                device float* mt           [[buffer(2)]],
                                device float* vt           [[buffer(3)]],
                                device const AdamWParams* p[[buffer(4)]],
                                uint id [[thread_position_in_grid]]) {
    float grad = g[id];

    float mNew = p->beta1 * mt[id] + (1.0f - p->beta1) * grad;
    float vNew = p->beta2 * vt[id] + (1.0f - p->beta2) * (grad * grad);

    mt[id] = mNew;
    vt[id] = vNew;

    float mHat = mNew / p->bc1;
    float vHat = vNew / p->bc2;

    float update = (p->lr * mHat) / (sqrt(vHat) + p->eps);
    if (p->weightDecay != 0.0f) {
        update += p->lr * p->weightDecay * w[id];
    }
    w[id] -= update;
}
)";

static void throwJavaRuntimeException(JNIEnv* env, const char* msg) {
    jclass exClass = env->FindClass("java/lang/RuntimeException");
    if (exClass != nullptr) {
        env->ThrowNew(exClass, msg);
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_nativeIsAvailable(
        JNIEnv*, jclass) {
    @autoreleasepool {
        return MTLCreateSystemDefaultDevice() == nil ? JNI_FALSE : JNI_TRUE;
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

static MTLCompileOptions* makeCompileOptions() {
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    if (@available(macOS 15.0, *)) {
        options.mathMode = MTLMathModeFast;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = YES;
#pragma clang diagnostic pop
    }
    return options;
}

static id<MTLLibrary> compileLibrary(id<MTLDevice> device) {
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:metalShaderSource
                                                  options:makeCompileOptions()
                                                    error:&error];
    if (library != nil) return library;
    NSString* desc = error.localizedDescription ?: @"Unknown error";
    throw std::runtime_error(std::string("Failed to compile Metal shaders: ") + [desc UTF8String]);
}

static void initBasicPipelines(MetalContext* ctx, id<MTLLibrary> library) {
    ctx->addPSO = makePSO(library, @"kernel_add");
    ctx->subtractPSO = makePSO(library, @"kernel_subtract");
    ctx->multiplyPSO = makePSO(library, @"kernel_multiply");
    ctx->dividePSO = makePSO(library, @"kernel_divide");
    ctx->multiplyScalarPSO = makePSO(library, @"kernel_multiply_scalar");
    ctx->addScalarPSO = makePSO(library, @"kernel_add_scalar");
    ctx->divideScalarPSO = makePSO(library, @"kernel_divide_scalar");
    ctx->transposePSO = makePSO(library, @"kernel_transpose");
}

static void initBroadcastPipelines(MetalContext* ctx, id<MTLLibrary> library) {
    ctx->addRowVectorPSO = makePSO(library, @"kernel_add_row_vector");
    ctx->addBroadcastColsPSO = makePSO(library, @"kernel_add_broadcast_cols");
    ctx->subtractBroadcastColsPSO = makePSO(library, @"kernel_subtract_broadcast_cols");
    ctx->divideBroadcastColsPSO = makePSO(library, @"kernel_divide_broadcast_cols");
    ctx->multiplyBroadcastColsPSO = makePSO(library, @"kernel_multiply_broadcast_cols");
    ctx->multiplyBroadcastRowsPSO = makePSO(library, @"kernel_multiply_broadcast_rows");
}

static void initReductionPipelines(MetalContext* ctx, id<MTLLibrary> library) {
    ctx->sumRowsPSO = makePSO(library, @"kernel_sum_rows");
    ctx->sumAlongRowsPSO = makePSO(library, @"kernel_sum_along_rows");
    ctx->meanAlongRowsPSO = makePSO(library, @"kernel_mean_along_rows");
    ctx->varianceAlongRowsPSO = makePSO(library, @"kernel_variance_along_rows");
    ctx->maxAlongRowsPSO = makePSO(library, @"kernel_max_along_rows");
    ctx->sumAbsPSO = makePSO(library, @"kernel_sum_abs");
}

static void initLossPipelines(MetalContext* ctx, id<MTLLibrary> library) {
    ctx->crossEntropyLossPSO = makePSO(library, @"kernel_cross_entropy_loss");
    ctx->crossEntropyGradPSO = makePSO(library, @"kernel_cross_entropy_gradient");
    ctx->clampPSO = makePSO(library, @"kernel_clamp");
    ctx->powPSO = makePSO(library, @"kernel_pow");
    ctx->scatterAddRowsPSO = makePSO(library, @"kernel_scatter_add_rows");
    ctx->scatterAddRowsAtomicPSO = makePSO(library, @"kernel_scatter_add_rows_atomic");
}

static void initUnaryPipelines(MetalContext* ctx, id<MTLLibrary> library) {
    ctx->sqrtPSO = makePSO(library, @"kernel_sqrt");
    ctx->negPSO = makePSO(library, @"kernel_neg");
    ctx->expPSO = makePSO(library, @"kernel_exp");
    ctx->logPSO = makePSO(library, @"kernel_log");
    ctx->tanhPSO = makePSO(library, @"kernel_tanh");
    ctx->sigmoidPSO = makePSO(library, @"kernel_sigmoid");
    ctx->reluPSO = makePSO(library, @"kernel_relu");
    ctx->reluBackwardPSO = makePSO(library, @"kernel_relu_backward");
    ctx->geluPSO = makePSO(library, @"kernel_gelu");
    ctx->geluBackwardPSO = makePSO(library, @"kernel_gelu_backward");
}

static void initTrainingPipelines(MetalContext* ctx, id<MTLLibrary> library) {
    ctx->softmaxMaxPSO = makePSO(library, @"kernel_softmax_max");
    ctx->softmaxExpSumPSO = makePSO(library, @"kernel_softmax_expsum");
    ctx->softmaxNormPSO = makePSO(library, @"kernel_softmax_norm");
    ctx->softmaxBackwardPSO = makePSO(library, @"kernel_softmax_backward");
    ctx->layerNormBackwardPSO = makePSO(library, @"kernel_layernorm_backward");
    ctx->adamWUpdatePSO = makePSO(library, @"kernel_adamw_update");
}

static MetalContext* createContext() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) throw std::runtime_error("Metal device not available");
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) throw std::runtime_error("Failed to create Metal command queue");
    id<MTLLibrary> library = compileLibrary(device);
    MetalContext* ctx = new MetalContext();
    ctx->device = device;
    ctx->queue = queue;
    ctx->library = library;
    initBasicPipelines(ctx, library);
    initBroadcastPipelines(ctx, library);
    initReductionPipelines(ctx, library);
    initLossPipelines(ctx, library);
    initUnaryPipelines(ctx, library);
    initTrainingPipelines(ctx, library);
    return ctx;
}

static MetalContext* getContext() {
    std::lock_guard<std::mutex> lock(gContextMutex);
    if (gCtx != nullptr) return gCtx;
    @autoreleasepool {
        gCtx = createContext();
    }
    return gCtx;
}

#include <unordered_map>

static std::unordered_map<int, id<MTLBuffer>> gBufferPool;
static std::mutex gBufferMutex;

static std::unordered_map<uint64_t, MPSMatrixMultiplication*> gMatmulKernelCache;

static MPSMatrixMultiplication* cachedMatmulKernel(MetalContext* ctx, int m, int n, int k) {

    uint64_t key = ((uint64_t)(uint32_t)m << 42) | ((uint64_t)(uint32_t)n << 21) | (uint64_t)(uint32_t)k;
    auto it = gMatmulKernelCache.find(key);
    if (it != gMatmulKernelCache.end()) return it->second;

    MPSMatrixMultiplication* mm =
        [[MPSMatrixMultiplication alloc] initWithDevice:ctx->device
            transposeLeft:NO transposeRight:NO
            resultRows:m resultColumns:n interiorColumns:k
            alpha:1.0 beta:0.0];
    gMatmulKernelCache[key] = mm;
    return mm;
}

// Keep these values synchronized with ComputeGraph.java.
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
static constexpr int OP_ADAMW_UPDATE   = 20;
static constexpr int OP_ADD_SCALAR     = 21;
static constexpr int OP_DIVIDE_SCALAR  = 22;
static constexpr int OP_TRANSPOSE      = 23;
static constexpr int OP_ADD_ROW_VECTOR = 24;
static constexpr int OP_ADD_BROADCAST_COLS = 25;
static constexpr int OP_SUBTRACT_BROADCAST_COLS = 26;
static constexpr int OP_DIVIDE_BROADCAST_COLS = 27;
static constexpr int OP_MULTIPLY_BROADCAST_ROWS = 28;
static constexpr int OP_SUM_ROWS       = 29;
static constexpr int OP_MEAN_ALONG_ROWS = 30;
static constexpr int OP_VARIANCE_ALONG_ROWS = 31;
static constexpr int OP_MULTIPLY_BROADCAST_COLS = 32;
static constexpr int OP_SUM_ALONG_ROWS = 33;
static constexpr int OP_MAX_ALONG_ROWS = 34;
static constexpr int OP_CLAMP = 35;
static constexpr int OP_POW = 36;
static constexpr int OP_SCATTER_ADD_ROWS = 37;
static constexpr int OP_SUM_ABS = 38;
static constexpr int OP_CROSS_ENTROPY_LOSS = 39;
static constexpr int OP_CROSS_ENTROPY_GRADIENT = 40;
static constexpr int OP_SUM_SCALAR = 41;
static constexpr int OP_SCATTER_ADD_ROWS_ATOMIC = 42;

static id<MTLBuffer> requireBuffer(int id, const char* opName) {
    auto it = gBufferPool.find(id);
    if (it == gBufferPool.end() || it->second == nil) {
        throw std::runtime_error(std::string(opName) + ": missing GPU buffer id=" + std::to_string(id));
    }
    return it->second;
}

static NSUInteger rowReductionWidth(id<MTLComputePipelineState> pso) {
    NSUInteger limit = MIN((NSUInteger)1024, pso.maxTotalThreadsPerThreadgroup);
    NSUInteger width = 1;
    while ((width << 1) <= limit && (width << 1) <= 256) {
        width <<= 1;
    }
    return width;
}

static NSUInteger colReductionHeight(id<MTLComputePipelineState> pso) {
    NSUInteger limit = MIN((NSUInteger)1024, pso.maxTotalThreadsPerThreadgroup);
    NSUInteger height = 1;
    while ((height << 1) <= limit && (height << 1) <= 256) {
        height <<= 1;
    }
    return height;
}

struct SoftmaxBuffers {
    id<MTLBuffer> maximum;
    id<MTLBuffer> sum;
    id<MTLBuffer> dimensions;
};

static SoftmaxBuffers makeSoftmaxBuffers(MetalContext* ctx, int rows, int cols) {
    NSUInteger bytes = (NSUInteger)rows * sizeof(float);
    id<MTLBuffer> maximum = [ctx->device newBufferWithLength:bytes
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> sum = [ctx->device newBufferWithLength:bytes
                                                 options:MTLResourceStorageModeShared];
    uint32_t width = (uint32_t)cols;
    id<MTLBuffer> dims = [ctx->device newBufferWithBytes:&width length:sizeof(width)
                                                  options:MTLResourceStorageModeShared];
    if (maximum == nil || sum == nil || dims == nil) {
        throw std::runtime_error("Failed to allocate softmax buffers");
    }
    return SoftmaxBuffers{maximum, sum, dims};
}

static void encodeSoftmaxMax(id<MTLComputeCommandEncoder> encoder, MetalContext* ctx,
                             id<MTLBuffer> input, const SoftmaxBuffers& buffers, int rows) {
    [encoder setComputePipelineState:ctx->softmaxMaxPSO];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:buffers.maximum offset:0 atIndex:1];
    [encoder setBuffer:buffers.dimensions offset:0 atIndex:2];
    NSUInteger threads = rowReductionWidth(ctx->softmaxMaxPSO);
    [encoder dispatchThreads:MTLSizeMake(threads, (NSUInteger)rows, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static void encodeSoftmaxExp(id<MTLComputeCommandEncoder> encoder, MetalContext* ctx,
                             id<MTLBuffer> input, id<MTLBuffer> output,
                             const SoftmaxBuffers& buffers, int rows) {
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:ctx->softmaxExpSumPSO];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBuffer:buffers.maximum offset:0 atIndex:2];
    [encoder setBuffer:buffers.sum offset:0 atIndex:3];
    [encoder setBuffer:buffers.dimensions offset:0 atIndex:4];
    NSUInteger threads = rowReductionWidth(ctx->softmaxExpSumPSO);
    [encoder dispatchThreads:MTLSizeMake(threads, (NSUInteger)rows, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static void encodeSoftmaxNorm(id<MTLComputeCommandEncoder> encoder, MetalContext* ctx,
                              id<MTLBuffer> output, const SoftmaxBuffers& buffers,
                              NSUInteger total) {
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:ctx->softmaxNormPSO];
    [encoder setBuffer:output offset:0 atIndex:0];
    [encoder setBuffer:buffers.sum offset:0 atIndex:1];
    [encoder setBuffer:buffers.dimensions offset:0 atIndex:2];
    NSUInteger threads = MIN(total, ctx->softmaxNormPSO.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static void encodeSoftmaxGraph(id<MTLComputeCommandEncoder> __strong &encoder,
                               id<MTLCommandBuffer> commandBuffer, MetalContext* ctx,
                               id<MTLBuffer> input, id<MTLBuffer> output,
                               int rows, int cols) {
    SoftmaxBuffers buffers = makeSoftmaxBuffers(ctx, rows, cols);
    if (encoder == nil) encoder = [commandBuffer computeCommandEncoder];
    encodeSoftmaxMax(encoder, ctx, input, buffers, rows);
    encodeSoftmaxExp(encoder, ctx, input, output, buffers, rows);
    encodeSoftmaxNorm(encoder, ctx, output, buffers,
                      (NSUInteger)rows * (NSUInteger)cols);
}

static bool validateAllocationArrays(JNIEnv* env, jintArray ids, jintArray sizes, jint count) {
    if (ids == nullptr || sizes == nullptr) {
        throwJavaRuntimeException(env, "GPU allocation arrays cannot be null");
        return false;
    }
    if (count >= 0 && count <= env->GetArrayLength(ids) &&
        count <= env->GetArrayLength(sizes)) return true;
    throwJavaRuntimeException(env, "Invalid GPU allocation count");
    return false;
}

static void allocateBuffer(MetalContext* ctx, int bufferId, int floatCount) {
    if (bufferId < 0 || floatCount <= 0) throw std::runtime_error("Invalid GPU buffer allocation");
    NSUInteger bytes = (NSUInteger)floatCount * sizeof(float);
    auto existing = gBufferPool.find(bufferId);
    if (existing != gBufferPool.end()) {
        if ([existing->second length] != bytes) {
            throw std::runtime_error("GPU buffer id already has a different size");
        }
        return;
    }
    id<MTLBuffer> buffer = [ctx->device newBufferWithLength:bytes
                                                    options:MTLResourceStorageModeShared];
    if (buffer == nil) throw std::runtime_error("Failed to allocate Metal buffer");
    gBufferPool[bufferId] = buffer;
}

static void allocateBuffers(const jint* ids, const jint* sizes, int count) {
    MetalContext* ctx = getContext();
    for (int i = 0; i < count; i++) allocateBuffer(ctx, ids[i], sizes[i]);
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_nativeAllocBuffers(
        JNIEnv* env, jclass, jintArray idsArr, jintArray sizesArr, jint count) {
    if (!validateAllocationArrays(env, idsArr, sizesArr, count)) return;
    jint* ids   = env->GetIntArrayElements(idsArr, nullptr);
    jint* sizes = env->GetIntArrayElements(sizesArr, nullptr);
    if (ids == nullptr || sizes == nullptr) {
        if (ids != nullptr) env->ReleaseIntArrayElements(idsArr, ids, JNI_ABORT);
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(gBufferMutex);
        allocateBuffers(ids, sizes, count);
    } catch (const std::exception& ex) {
        env->ReleaseIntArrayElements(idsArr, ids, JNI_ABORT);
        env->ReleaseIntArrayElements(sizesArr, sizes, JNI_ABORT);
        throwJavaRuntimeException(env, ex.what());
        return;
    }
    env->ReleaseIntArrayElements(idsArr, ids, JNI_ABORT);
    env->ReleaseIntArrayElements(sizesArr, sizes, JNI_ABORT);
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_nativeUploadBuffer(
        JNIEnv* env, jclass, jint bufId, jfloatArray dataArr) {
    if (dataArr == nullptr) {
        throwJavaRuntimeException(env, "Upload data cannot be null");
        return;
    }
    std::lock_guard<std::mutex> lock(gBufferMutex);
    auto it = gBufferPool.find(bufId);
    if (it == gBufferPool.end()) {
        throwJavaRuntimeException(env, "Buffer not found for upload");
        return;
    }
    jint len = env->GetArrayLength(dataArr);
    if ((NSUInteger)len * sizeof(float) != [it->second length]) {
        throwJavaRuntimeException(env, "Upload length does not match Metal buffer size");
        return;
    }
    jfloat* data = env->GetFloatArrayElements(dataArr, nullptr);
    if (data == nullptr) return;
    std::memcpy([it->second contents], data, (size_t)len * sizeof(float));
    env->ReleaseFloatArrayElements(dataArr, data, JNI_ABORT);
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_nativeDownloadBuffer(
        JNIEnv* env, jclass, jint bufId, jfloatArray outArr) {
    if (outArr == nullptr) {
        throwJavaRuntimeException(env, "Download target cannot be null");
        return;
    }
    std::lock_guard<std::mutex> lock(gBufferMutex);
    auto it = gBufferPool.find(bufId);
    if (it == gBufferPool.end()) {
        throwJavaRuntimeException(env, "Buffer not found for download");
        return;
    }
    jint len = env->GetArrayLength(outArr);
    if ((NSUInteger)len * sizeof(float) != [it->second length]) {
        throwJavaRuntimeException(env, "Download length does not match Metal buffer size");
        return;
    }
    jfloat* out = env->GetFloatArrayElements(outArr, nullptr);
    if (out == nullptr) return;
    std::memcpy(out, [it->second contents], (size_t)len * sizeof(float));
    env->ReleaseFloatArrayElements(outArr, out, 0);
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_nativeReleaseBuffers(
        JNIEnv* env, jclass, jintArray idsArr, jint count) {
    if (idsArr == nullptr || count < 0 || count > env->GetArrayLength(idsArr)) {
        throwJavaRuntimeException(env, "Invalid GPU release request");
        return;
    }
    std::lock_guard<std::mutex> lock(gBufferMutex);
    jint* ids = env->GetIntArrayElements(idsArr, nullptr);
    if (ids == nullptr) return;
    for (int i = 0; i < count; i++) {
        gBufferPool.erase(ids[i]);
    }
    env->ReleaseIntArrayElements(idsArr, ids, JNI_ABORT);
}

struct GraphState {
    MetalContext* ctx;
    id<MTLCommandBuffer> commandBuffer;
    id<MTLComputeCommandEncoder> encoder;
};

struct AdamWParamsHost {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weightDecay;
    float bc1;
    float bc2;
};

static id<MTLComputeCommandEncoder> graphEncoder(GraphState& state) {
    if (state.encoder == nil) {
        state.encoder = [state.commandBuffer computeCommandEncoder];
    }
    return state.encoder;
}

static id<MTLBuffer> valueBuffer(MetalContext* ctx, const void* value, NSUInteger bytes) {
    id<MTLBuffer> buffer = [ctx->device newBufferWithBytes:value
                                                   length:bytes
                                                  options:MTLResourceStorageModeShared];
    if (buffer == nil) throw std::runtime_error("Failed to allocate Metal value buffer");
    return buffer;
}

static float floatFromBits(int bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(float));
    return value;
}

static NSUInteger positiveCount(int value, const char* name) {
    if (value <= 0) throw std::runtime_error(std::string(name) + " must be positive");
    return (NSUInteger)value;
}

static NSUInteger elementCount(int rows, int cols) {
    return positiveCount(rows, "rows") * positiveCount(cols, "cols");
}

static void dispatch1D(id<MTLComputeCommandEncoder> encoder,
                       id<MTLComputePipelineState> pipeline, NSUInteger count) {
    NSUInteger threads = MIN(count, pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static id<MTLComputePipelineState> binaryPipeline(MetalContext* ctx, int op) {
    switch (op) {
        case OP_ADD: return ctx->addPSO;
        case OP_SUBTRACT: return ctx->subtractPSO;
        case OP_MULTIPLY: return ctx->multiplyPSO;
        case OP_DIVIDE: return ctx->dividePSO;
        case OP_RELU_BACKWARD: return ctx->reluBackwardPSO;
        case OP_GELU_BACKWARD: return ctx->geluBackwardPSO;
        default: throw std::runtime_error("Invalid binary op");
    }
}

static id<MTLComputePipelineState> unaryPipeline(MetalContext* ctx, int op) {
    switch (op) {
        case OP_SQRT: return ctx->sqrtPSO;
        case OP_NEG: return ctx->negPSO;
        case OP_EXP: return ctx->expPSO;
        case OP_LOG: return ctx->logPSO;
        case OP_TANH: return ctx->tanhPSO;
        case OP_SIGMOID: return ctx->sigmoidPSO;
        case OP_RELU: return ctx->reluPSO;
        case OP_GELU: return ctx->geluPSO;
        default: throw std::runtime_error("Invalid unary op");
    }
}

static id<MTLComputePipelineState> scalarPipeline(MetalContext* ctx, int op) {
    switch (op) {
        case OP_MULTIPLY_SCALAR: return ctx->multiplyScalarPSO;
        case OP_ADD_SCALAR: return ctx->addScalarPSO;
        case OP_DIVIDE_SCALAR: return ctx->divideScalarPSO;
        case OP_POW: return ctx->powPSO;
        default: throw std::runtime_error("Invalid scalar op");
    }
}

static void encodeBinary(GraphState& state, const jint* cmd, int pos) {
    id<MTLComputePipelineState> pipeline = binaryPipeline(state.ctx, cmd[pos]);
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "binary op") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "binary op") offset:0 atIndex:1];
    [encoder setBuffer:requireBuffer(cmd[pos + 3], "binary op") offset:0 atIndex:2];
    dispatch1D(encoder, pipeline, positiveCount(cmd[pos + 4], "element count"));
}

static void encodeUnary(GraphState& state, const jint* cmd, int pos) {
    id<MTLComputePipelineState> pipeline = unaryPipeline(state.ctx, cmd[pos]);
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "unary op") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "unary op") offset:0 atIndex:1];
    dispatch1D(encoder, pipeline, positiveCount(cmd[pos + 3], "element count"));
}

static void encodeScalar(GraphState& state, const jint* cmd, int pos) {
    float scalar = floatFromBits(cmd[pos + 3]);
    id<MTLBuffer> value = valueBuffer(state.ctx, &scalar, sizeof(float));
    id<MTLComputePipelineState> pipeline = scalarPipeline(state.ctx, cmd[pos]);
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "scalar op") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "scalar op") offset:0 atIndex:1];
    [encoder setBuffer:value offset:0 atIndex:2];
    dispatch1D(encoder, pipeline, positiveCount(cmd[pos + 4], "element count"));
}

static void encodeClamp(GraphState& state, const jint* cmd, int pos) {
    float range[] = {floatFromBits(cmd[pos + 3]), floatFromBits(cmd[pos + 4])};
    id<MTLBuffer> rangeBuffer = valueBuffer(state.ctx, range, sizeof(range));
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->clampPSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "clamp op") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "clamp op") offset:0 atIndex:1];
    [encoder setBuffer:rangeBuffer offset:0 atIndex:2];
    dispatch1D(encoder, state.ctx->clampPSO,
               positiveCount(cmd[pos + 5], "element count"));
}

static void encodeTranspose(GraphState& state, const jint* cmd, int pos) {
    uint32_t dims[] = {(uint32_t)cmd[pos + 3], (uint32_t)cmd[pos + 4]};
    NSUInteger total = elementCount(cmd[pos + 3], cmd[pos + 4]);
    id<MTLBuffer> dimsBuffer = valueBuffer(state.ctx, dims, sizeof(dims));
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->transposePSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "transpose") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "transpose") offset:0 atIndex:1];
    [encoder setBuffer:dimsBuffer offset:0 atIndex:2];
    dispatch1D(encoder, state.ctx->transposePSO, total);
}

static id<MTLComputePipelineState> scatterPipeline(MetalContext* ctx, int op) {
    if (op == OP_SCATTER_ADD_ROWS) return ctx->scatterAddRowsPSO;
    if (op == OP_SCATTER_ADD_ROWS_ATOMIC) return ctx->scatterAddRowsAtomicPSO;
    throw std::runtime_error("Invalid scatter op");
}

static void encodeScatter(GraphState& state, const jint* cmd, int pos) {
    uint32_t dims[] = {(uint32_t)cmd[pos + 4], (uint32_t)cmd[pos + 5],
                       (uint32_t)cmd[pos + 6]};
    NSUInteger total = elementCount(cmd[pos + 5], cmd[pos + 6]);
    id<MTLBuffer> dimsBuffer = valueBuffer(state.ctx, dims, sizeof(dims));
    id<MTLComputePipelineState> pipeline = scatterPipeline(state.ctx, cmd[pos]);
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "scatter") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "scatter") offset:0 atIndex:1];
    [encoder setBuffer:requireBuffer(cmd[pos + 3], "scatter") offset:0 atIndex:2];
    [encoder setBuffer:dimsBuffer offset:0 atIndex:3];
    dispatch1D(encoder, pipeline, total);
}

static id<MTLComputePipelineState> broadcastPipeline(MetalContext* ctx, int op) {
    switch (op) {
        case OP_ADD_ROW_VECTOR: return ctx->addRowVectorPSO;
        case OP_ADD_BROADCAST_COLS: return ctx->addBroadcastColsPSO;
        case OP_SUBTRACT_BROADCAST_COLS: return ctx->subtractBroadcastColsPSO;
        case OP_DIVIDE_BROADCAST_COLS: return ctx->divideBroadcastColsPSO;
        case OP_MULTIPLY_BROADCAST_ROWS: return ctx->multiplyBroadcastRowsPSO;
        case OP_MULTIPLY_BROADCAST_COLS: return ctx->multiplyBroadcastColsPSO;
        default: throw std::runtime_error("Invalid broadcast op");
    }
}

static void encodeBroadcast(GraphState& state, const jint* cmd, int pos) {
    uint32_t dims[] = {(uint32_t)cmd[pos + 4], (uint32_t)cmd[pos + 5]};
    NSUInteger total = elementCount(cmd[pos + 4], cmd[pos + 5]);
    id<MTLBuffer> dimsBuffer = valueBuffer(state.ctx, dims, sizeof(dims));
    id<MTLComputePipelineState> pipeline = broadcastPipeline(state.ctx, cmd[pos]);
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "broadcast") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "broadcast") offset:0 atIndex:1];
    [encoder setBuffer:requireBuffer(cmd[pos + 3], "broadcast") offset:0 atIndex:2];
    [encoder setBuffer:dimsBuffer offset:0 atIndex:3];
    dispatch1D(encoder, pipeline, total);
}

static id<MTLComputePipelineState> reductionPipeline(MetalContext* ctx, int op) {
    switch (op) {
        case OP_SUM_ROWS: return ctx->sumRowsPSO;
        case OP_SUM_ALONG_ROWS: return ctx->sumAlongRowsPSO;
        case OP_MEAN_ALONG_ROWS: return ctx->meanAlongRowsPSO;
        case OP_VARIANCE_ALONG_ROWS: return ctx->varianceAlongRowsPSO;
        case OP_MAX_ALONG_ROWS: return ctx->maxAlongRowsPSO;
        default: throw std::runtime_error("Invalid reduction op");
    }
}

static void dispatchReduction(id<MTLComputeCommandEncoder> encoder,
                              id<MTLComputePipelineState> pipeline,
                              int op, NSUInteger rows, NSUInteger cols) {
    if (op != OP_SUM_ROWS) {
        NSUInteger threads = rowReductionWidth(pipeline);
        [encoder dispatchThreads:MTLSizeMake(threads, rows, 1)
           threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        return;
    }
    NSUInteger threads = colReductionHeight(pipeline);
    [encoder dispatchThreads:MTLSizeMake(cols, threads, 1)
       threadsPerThreadgroup:MTLSizeMake(1, threads, 1)];
}

static void encodeReduction(GraphState& state, const jint* cmd, int pos) {
    NSUInteger rows = positiveCount(cmd[pos + 3], "rows");
    NSUInteger cols = positiveCount(cmd[pos + 4], "cols");
    uint32_t dims[] = {(uint32_t)rows, (uint32_t)cols};
    id<MTLBuffer> dimsBuffer = valueBuffer(state.ctx, dims, sizeof(dims));
    id<MTLComputePipelineState> pipeline = reductionPipeline(state.ctx, cmd[pos]);
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "reduction") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "reduction") offset:0 atIndex:1];
    [encoder setBuffer:dimsBuffer offset:0 atIndex:2];
    dispatchReduction(encoder, pipeline, cmd[pos], rows, cols);
}

static void encodeSumAbs(GraphState& state, const jint* cmd, int pos) {
    NSUInteger rows = positiveCount(cmd[pos + 3], "rows");
    uint32_t dims[] = {(uint32_t)rows, (uint32_t)positiveCount(cmd[pos + 4], "cols")};
    id<MTLBuffer> dimsBuffer = valueBuffer(state.ctx, dims, sizeof(dims));
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->sumAbsPSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "sum abs") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "sum abs") offset:0 atIndex:1];
    [encoder setBuffer:dimsBuffer offset:0 atIndex:2];
    NSUInteger threads = MIN((NSUInteger)256,
                             state.ctx->sumAbsPSO.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(threads, rows, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static void encodeColumnSum(GraphState& state, id<MTLBuffer> input,
                            id<MTLBuffer> output, id<MTLBuffer> dims, NSUInteger cols) {
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->sumRowsPSO];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBuffer:dims offset:0 atIndex:2];
    NSUInteger threads = colReductionHeight(state.ctx->sumRowsPSO);
    [encoder dispatchThreads:MTLSizeMake(cols, threads, 1)
       threadsPerThreadgroup:MTLSizeMake(1, threads, 1)];
}

static void encodeFinalSum(GraphState& state, id<MTLBuffer> input,
                           id<MTLBuffer> output, id<MTLBuffer> dims) {
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:state.ctx->sumAlongRowsPSO];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBuffer:dims offset:0 atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
}

static void encodeScalarSum(GraphState& state, const jint* cmd, int pos) {
    NSUInteger rows = positiveCount(cmd[pos + 3], "rows");
    NSUInteger cols = positiveCount(cmd[pos + 4], "cols");
    uint32_t inputDims[] = {(uint32_t)rows, (uint32_t)cols};
    uint32_t outputDims[] = {1u, (uint32_t)cols};
    id<MTLBuffer> temporary = [state.ctx->device
        newBufferWithLength:cols * sizeof(float) options:MTLResourceStorageModeShared];
    if (temporary == nil) throw std::runtime_error("Failed to allocate sum buffer");
    id<MTLBuffer> inputDimsBuffer = valueBuffer(state.ctx, inputDims, sizeof(inputDims));
    id<MTLBuffer> outputDimsBuffer = valueBuffer(state.ctx, outputDims, sizeof(outputDims));
    encodeColumnSum(state, requireBuffer(cmd[pos + 1], "sum"), temporary,
                    inputDimsBuffer, cols);
    encodeFinalSum(state, temporary, requireBuffer(cmd[pos + 2], "sum"),
                   outputDimsBuffer);
}

static id<MTLComputePipelineState> crossEntropyPipeline(MetalContext* ctx, int op) {
    if (op == OP_CROSS_ENTROPY_LOSS) return ctx->crossEntropyLossPSO;
    if (op == OP_CROSS_ENTROPY_GRADIENT) return ctx->crossEntropyGradPSO;
    throw std::runtime_error("Invalid cross entropy op");
}

static void encodeCrossEntropy(GraphState& state, const jint* cmd, int pos) {
    NSUInteger rows = positiveCount(cmd[pos + 4], "rows");
    uint32_t dims[] = {(uint32_t)rows, (uint32_t)positiveCount(cmd[pos + 5], "cols")};
    id<MTLBuffer> dimsBuffer = valueBuffer(state.ctx, dims, sizeof(dims));
    id<MTLComputePipelineState> pipeline = crossEntropyPipeline(state.ctx, cmd[pos]);
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "cross entropy") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "cross entropy") offset:0 atIndex:1];
    [encoder setBuffer:requireBuffer(cmd[pos + 3], "cross entropy") offset:0 atIndex:2];
    [encoder setBuffer:dimsBuffer offset:0 atIndex:3];
    NSUInteger threads = rowReductionWidth(pipeline);
    [encoder dispatchThreads:MTLSizeMake(threads, rows, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static MPSMatrixDescriptor* matrixDescriptor(int rows, int cols) {
    positiveCount(rows, "matrix rows");
    positiveCount(cols, "matrix cols");
    return [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols
        rowBytes:(NSUInteger)cols * sizeof(float) dataType:MPSDataTypeFloat32];
}

static void encodeMatmul(GraphState& state, const jint* cmd, int pos) {
    if (state.encoder != nil) {
        [state.encoder endEncoding];
        state.encoder = nil;
    }
    int m = cmd[pos + 4], n = cmd[pos + 5], k = cmd[pos + 6];
    MPSMatrix* a = [[MPSMatrix alloc]
        initWithBuffer:requireBuffer(cmd[pos + 1], "matmul")
             descriptor:matrixDescriptor(m, k)];
    MPSMatrix* b = [[MPSMatrix alloc]
        initWithBuffer:requireBuffer(cmd[pos + 2], "matmul")
             descriptor:matrixDescriptor(k, n)];
    MPSMatrix* out = [[MPSMatrix alloc]
        initWithBuffer:requireBuffer(cmd[pos + 3], "matmul")
             descriptor:matrixDescriptor(m, n)];
    [cachedMatmulKernel(state.ctx, m, n, k)
        encodeToCommandBuffer:state.commandBuffer leftMatrix:a rightMatrix:b resultMatrix:out];
}

static void encodeSoftmax(GraphState& state, const jint* cmd, int pos) {
    positiveCount(cmd[pos + 3], "rows");
    positiveCount(cmd[pos + 4], "cols");
    encodeSoftmaxGraph(state.encoder, state.commandBuffer, state.ctx,
                       requireBuffer(cmd[pos + 1], "softmax"),
                       requireBuffer(cmd[pos + 2], "softmax"),
                       cmd[pos + 3], cmd[pos + 4]);
}

static void encodeSoftmaxBackward(GraphState& state, const jint* cmd, int pos) {
    NSUInteger rows = positiveCount(cmd[pos + 4], "rows");
    uint32_t cols = (uint32_t)positiveCount(cmd[pos + 5], "cols");
    id<MTLBuffer> dimsBuffer = valueBuffer(state.ctx, &cols, sizeof(cols));
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->softmaxBackwardPSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "softmax backward") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "softmax backward") offset:0 atIndex:1];
    [encoder setBuffer:requireBuffer(cmd[pos + 3], "softmax backward") offset:0 atIndex:2];
    [encoder setBuffer:dimsBuffer offset:0 atIndex:3];
    NSUInteger threads = rowReductionWidth(state.ctx->softmaxBackwardPSO);
    [encoder dispatchThreads:MTLSizeMake(threads, rows, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static void encodeLayerNormBackward(GraphState& state, const jint* cmd, int pos) {
    NSUInteger rows = positiveCount(cmd[pos + 5], "rows");
    uint32_t cols = (uint32_t)positiveCount(cmd[pos + 6], "cols");
    id<MTLBuffer> dimsBuffer = valueBuffer(state.ctx, &cols, sizeof(cols));
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->layerNormBackwardPSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "layer norm backward") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "layer norm backward") offset:0 atIndex:1];
    [encoder setBuffer:requireBuffer(cmd[pos + 3], "layer norm backward") offset:0 atIndex:2];
    [encoder setBuffer:requireBuffer(cmd[pos + 4], "layer norm backward") offset:0 atIndex:3];
    [encoder setBuffer:dimsBuffer offset:0 atIndex:4];
    NSUInteger threads = rowReductionWidth(state.ctx->layerNormBackwardPSO);
    [encoder dispatchThreads:MTLSizeMake(threads, rows, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static AdamWParamsHost adamWParams(const jint* cmd, int pos) {
    return AdamWParamsHost{
        floatFromBits(cmd[pos + 5]),
        floatFromBits(cmd[pos + 6]),
        floatFromBits(cmd[pos + 7]),
        floatFromBits(cmd[pos + 8]),
        floatFromBits(cmd[pos + 9]),
        floatFromBits(cmd[pos + 10]),
        floatFromBits(cmd[pos + 11])
    };
}

static void encodeAdamW(GraphState& state, const jint* cmd, int pos) {
    AdamWParamsHost params = adamWParams(cmd, pos);
    id<MTLBuffer> paramsBuffer = valueBuffer(state.ctx, &params, sizeof(params));
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->adamWUpdatePSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "AdamW weights") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "AdamW gradient") offset:0 atIndex:1];
    [encoder setBuffer:requireBuffer(cmd[pos + 3], "AdamW first moment") offset:0 atIndex:2];
    [encoder setBuffer:requireBuffer(cmd[pos + 4], "AdamW second moment") offset:0 atIndex:3];
    [encoder setBuffer:paramsBuffer offset:0 atIndex:4];
    dispatch1D(encoder, state.ctx->adamWUpdatePSO,
               positiveCount(cmd[pos + 12], "element count"));
}

static bool isBinaryOp(int op) {
    return op == OP_ADD || op == OP_SUBTRACT || op == OP_MULTIPLY ||
           op == OP_DIVIDE || op == OP_RELU_BACKWARD || op == OP_GELU_BACKWARD;
}

static bool isUnaryOp(int op) {
    return op == OP_SQRT || op == OP_NEG || op == OP_EXP || op == OP_LOG ||
           op == OP_TANH || op == OP_SIGMOID || op == OP_RELU || op == OP_GELU;
}

static bool isScalarOp(int op) {
    return op == OP_MULTIPLY_SCALAR || op == OP_ADD_SCALAR ||
           op == OP_DIVIDE_SCALAR || op == OP_POW;
}

static bool isBroadcastOp(int op) {
    return op == OP_ADD_ROW_VECTOR || op == OP_ADD_BROADCAST_COLS ||
           op == OP_SUBTRACT_BROADCAST_COLS || op == OP_DIVIDE_BROADCAST_COLS ||
           op == OP_MULTIPLY_BROADCAST_ROWS || op == OP_MULTIPLY_BROADCAST_COLS;
}

static bool isReductionOp(int op) {
    return op == OP_SUM_ROWS || op == OP_SUM_ALONG_ROWS ||
           op == OP_MEAN_ALONG_ROWS || op == OP_VARIANCE_ALONG_ROWS ||
           op == OP_MAX_ALONG_ROWS;
}

static int commandWidth(int op) {
    if (isUnaryOp(op)) return 4;
    if (isBinaryOp(op) || isScalarOp(op) || isReductionOp(op)) return 5;
    if (op == OP_TRANSPOSE || op == OP_SOFTMAX_ROWS ||
        op == OP_SUM_ABS || op == OP_SUM_SCALAR) return 5;
    if (isBroadcastOp(op) || op == OP_CLAMP ||
        op == OP_SOFTMAX_BACKWARD || op == OP_CROSS_ENTROPY_LOSS ||
        op == OP_CROSS_ENTROPY_GRADIENT) return 6;
    if (op == OP_MATMUL || op == OP_SCATTER_ADD_ROWS ||
        op == OP_SCATTER_ADD_ROWS_ATOMIC || op == OP_LAYERNORM_BACKWARD) return 7;
    if (op == OP_ADAMW_UPDATE) return 13;
    throw std::runtime_error("Unknown op code in graph: " + std::to_string(op));
}

static void encodeCommand(GraphState& state, const jint* cmd, int pos) {
    int op = cmd[pos];
    if (isBinaryOp(op)) encodeBinary(state, cmd, pos);
    else if (isUnaryOp(op)) encodeUnary(state, cmd, pos);
    else if (isScalarOp(op)) encodeScalar(state, cmd, pos);
    else if (isBroadcastOp(op)) encodeBroadcast(state, cmd, pos);
    else if (isReductionOp(op)) encodeReduction(state, cmd, pos);
    else if (op == OP_CLAMP) encodeClamp(state, cmd, pos);
    else if (op == OP_TRANSPOSE) encodeTranspose(state, cmd, pos);
    else if (op == OP_SCATTER_ADD_ROWS || op == OP_SCATTER_ADD_ROWS_ATOMIC) encodeScatter(state, cmd, pos);
    else if (op == OP_SUM_ABS) encodeSumAbs(state, cmd, pos);
    else if (op == OP_SUM_SCALAR) encodeScalarSum(state, cmd, pos);
    else if (op == OP_CROSS_ENTROPY_LOSS || op == OP_CROSS_ENTROPY_GRADIENT) encodeCrossEntropy(state, cmd, pos);
    else if (op == OP_MATMUL) encodeMatmul(state, cmd, pos);
    else if (op == OP_SOFTMAX_ROWS) encodeSoftmax(state, cmd, pos);
    else if (op == OP_SOFTMAX_BACKWARD) encodeSoftmaxBackward(state, cmd, pos);
    else if (op == OP_LAYERNORM_BACKWARD) encodeLayerNormBackward(state, cmd, pos);
    else if (op == OP_ADAMW_UPDATE) encodeAdamW(state, cmd, pos);
    else throw std::runtime_error("Unknown op code in graph: " + std::to_string(op));
}

static void encodeCommandStream(GraphState& state, const jint* cmd, int length) {
    int pos = 0;
    while (pos < length) {
        int width = commandWidth(cmd[pos]);
        if (width > length - pos) throw std::runtime_error("Truncated Metal command stream");
        encodeCommand(state, cmd, pos);
        pos += width;
    }
}

static void finishCommandBuffer(GraphState& state) {
    if (state.encoder != nil) [state.encoder endEncoding];
    [state.commandBuffer commit];
    [state.commandBuffer waitUntilCompleted];
    if (state.commandBuffer.status != MTLCommandBufferStatusError) return;
    NSString* desc = state.commandBuffer.error.localizedDescription ?: @"Unknown error";
    throw std::runtime_error(std::string("Metal command buffer error: ") + [desc UTF8String]);
}

static void executeGraph(const jint* commandStream, int length) {
    @autoreleasepool {
        MetalContext* ctx = getContext();
        id<MTLCommandBuffer> commandBuffer = [ctx->queue commandBuffer];
        if (commandBuffer == nil) throw std::runtime_error("Failed to create Metal command buffer");
        GraphState state{ctx, commandBuffer, nil};
        encodeCommandStream(state, commandStream, length);
        finishCommandBuffer(state);
    }
}

static bool validateCommandArray(JNIEnv* env, jintArray commands, jint length) {
    if (commands == nullptr) {
        throwJavaRuntimeException(env, "Metal command stream cannot be null");
        return false;
    }
    jsize arrayLength = env->GetArrayLength(commands);
    if (length >= 0 && length <= arrayLength) return true;
    throwJavaRuntimeException(env, "Invalid Metal command stream length");
    return false;
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_nativeFlushOps(
        JNIEnv* env, jclass, jintArray cmdStreamArr, jint cmdStreamLength) {
    if (!validateCommandArray(env, cmdStreamArr, cmdStreamLength)) return;
    jint* cmd = env->GetIntArrayElements(cmdStreamArr, nullptr);
    if (cmd == nullptr) return;
    try {
        std::lock_guard<std::mutex> lock(gBufferMutex);
        executeGraph(cmd, cmdStreamLength);
    } catch (const std::exception& ex) {
        env->ReleaseIntArrayElements(cmdStreamArr, cmd, JNI_ABORT);
        throwJavaRuntimeException(env, ex.what());
        return;
    }
    env->ReleaseIntArrayElements(cmdStreamArr, cmd, JNI_ABORT);
}
