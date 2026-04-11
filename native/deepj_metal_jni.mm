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
        gCtx->addScalarPSO      = makePSO(library, @"kernel_add_scalar");
        gCtx->divideScalarPSO   = makePSO(library, @"kernel_divide_scalar");
        gCtx->transposePSO      = makePSO(library, @"kernel_transpose");
        gCtx->addRowVectorPSO   = makePSO(library, @"kernel_add_row_vector");
        gCtx->addBroadcastColsPSO = makePSO(library, @"kernel_add_broadcast_cols");
        gCtx->subtractBroadcastColsPSO = makePSO(library, @"kernel_subtract_broadcast_cols");
        gCtx->divideBroadcastColsPSO = makePSO(library, @"kernel_divide_broadcast_cols");
        gCtx->multiplyBroadcastColsPSO = makePSO(library, @"kernel_multiply_broadcast_cols");
        gCtx->multiplyBroadcastRowsPSO = makePSO(library, @"kernel_multiply_broadcast_rows");
        gCtx->sumRowsPSO        = makePSO(library, @"kernel_sum_rows");
        gCtx->sumAlongRowsPSO   = makePSO(library, @"kernel_sum_along_rows");
        gCtx->meanAlongRowsPSO  = makePSO(library, @"kernel_mean_along_rows");
        gCtx->varianceAlongRowsPSO = makePSO(library, @"kernel_variance_along_rows");
        gCtx->maxAlongRowsPSO   = makePSO(library, @"kernel_max_along_rows");
        gCtx->sumAbsPSO         = makePSO(library, @"kernel_sum_abs");
        gCtx->crossEntropyLossPSO = makePSO(library, @"kernel_cross_entropy_loss");
        gCtx->crossEntropyGradPSO = makePSO(library, @"kernel_cross_entropy_gradient");
        gCtx->clampPSO          = makePSO(library, @"kernel_clamp");
        gCtx->powPSO            = makePSO(library, @"kernel_pow");
        gCtx->scatterAddRowsPSO = makePSO(library, @"kernel_scatter_add_rows");
        gCtx->scatterAddRowsAtomicPSO = makePSO(library, @"kernel_scatter_add_rows_atomic");
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
        gCtx->adamWUpdatePSO    = makePSO(library, @"kernel_adamw_update");
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
    NSUInteger tpg1 = rowReductionWidth(ctx->softmaxMaxPSO);
    [enc dispatchThreads:MTLSizeMake(tpg1, (NSUInteger)rows, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg1, 1, 1)];

    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Pass 2: exp + sum per row
    [enc setComputePipelineState:ctx->softmaxExpSumPSO];
    [enc setBuffer:bufIn   offset:0 atIndex:0];
    [enc setBuffer:bufOut  offset:0 atIndex:1];
    [enc setBuffer:bufMax  offset:0 atIndex:2];
    [enc setBuffer:bufSum  offset:0 atIndex:3];
    [enc setBuffer:bufDims offset:0 atIndex:4];
    NSUInteger tpg2 = rowReductionWidth(ctx->softmaxExpSumPSO);
    [enc dispatchThreads:MTLSizeMake(tpg2, (NSUInteger)rows, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg2, 1, 1)];

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

                // ── Scalar multiply/add/divide: [op, in, out, scalarBits, n] ────
                case OP_MULTIPLY_SCALAR:
                case OP_ADD_SCALAR:
                case OP_DIVIDE_SCALAR:
                case OP_POW: {
                    int inId = cmd[pos+1], outId = cmd[pos+2];
                    float scalar;
                    int bits = cmd[pos+3];
                    std::memcpy(&scalar, &bits, sizeof(float));
                    int n = cmd[pos+4];

                    id<MTLBuffer> scalarBuf = [ctx->device newBufferWithBytes:&scalar
                                               length:sizeof(float)
                                               options:MTLResourceStorageModeShared];

                    id<MTLComputePipelineState> pso;
                    switch (opCode) {
                        case OP_MULTIPLY_SCALAR: pso = ctx->multiplyScalarPSO; break;
                        case OP_ADD_SCALAR:      pso = ctx->addScalarPSO; break;
                        case OP_DIVIDE_SCALAR:   pso = ctx->divideScalarPSO; break;
                        case OP_POW:             pso = ctx->powPSO; break;
                        default: pso = ctx->multiplyScalarPSO; break;
                    }

                    if (!enc) enc = [cmdBuf computeCommandEncoder];
                    [enc setComputePipelineState:pso];
                    [enc setBuffer:gBufferPool[inId]  offset:0 atIndex:0];
                    [enc setBuffer:gBufferPool[outId] offset:0 atIndex:1];
                    [enc setBuffer:scalarBuf          offset:0 atIndex:2];
                    NSUInteger tpg = MIN((NSUInteger)n, pso.maxTotalThreadsPerThreadgroup);
                    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
                    pos += 5;
                    break;
                }

                // ── Clamp: [op, in, out, minBits, maxBits, n] ───────────
                case OP_CLAMP: {
                    int inId = cmd[pos+1], outId = cmd[pos+2];
                    float minV, maxV;
                    int minBits = cmd[pos+3], maxBits = cmd[pos+4];
                    std::memcpy(&minV, &minBits, sizeof(float));
                    std::memcpy(&maxV, &maxBits, sizeof(float));
                    int n = cmd[pos+5];

                    float minMax[2] = { minV, maxV };
                    id<MTLBuffer> minMaxBuf = [ctx->device newBufferWithBytes:minMax
                                               length:sizeof(minMax)
                                               options:MTLResourceStorageModeShared];

                    if (!enc) enc = [cmdBuf computeCommandEncoder];
                    [enc setComputePipelineState:ctx->clampPSO];
                    [enc setBuffer:gBufferPool[inId]  offset:0 atIndex:0];
                    [enc setBuffer:gBufferPool[outId] offset:0 atIndex:1];
                    [enc setBuffer:minMaxBuf          offset:0 atIndex:2];
                    NSUInteger tpg = MIN((NSUInteger)n, ctx->clampPSO.maxTotalThreadsPerThreadgroup);
                    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
                    pos += 6;
                    break;
                }

                // ── Transpose: [op, in, out, rows, cols] ─────────────
                case OP_TRANSPOSE: {
                    int inId = cmd[pos+1], outId = cmd[pos+2];
                    uint32_t rows = (uint32_t)cmd[pos+3];
                    uint32_t cols = (uint32_t)cmd[pos+4];
                    uint32_t dims[2] = { rows, cols };
                    id<MTLBuffer> dimsBuf = [ctx->device newBufferWithBytes:dims
                                             length:sizeof(dims)
                                             options:MTLResourceStorageModeShared];

                    NSUInteger total = (NSUInteger)rows * (NSUInteger)cols;
                    if (!enc) enc = [cmdBuf computeCommandEncoder];
                    [enc setComputePipelineState:ctx->transposePSO];
                    [enc setBuffer:gBufferPool[inId]  offset:0 atIndex:0];
                    [enc setBuffer:gBufferPool[outId] offset:0 atIndex:1];
                    [enc setBuffer:dimsBuf            offset:0 atIndex:2];
                    NSUInteger tpg = MIN(total, ctx->transposePSO.maxTotalThreadsPerThreadgroup);
                    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
                    pos += 5;
                    break;
                }

                // ── Scatter add rows (in-place): [op, target, indices, grad, targetRows, targetCols, nIdx] ──
                case OP_SCATTER_ADD_ROWS: {
                    int targetId = cmd[pos+1], indicesId = cmd[pos+2], gradId = cmd[pos+3];
                    uint32_t targetRows = (uint32_t)cmd[pos+4];
                    uint32_t targetCols = (uint32_t)cmd[pos+5];
                    uint32_t nIdx = (uint32_t)cmd[pos+6];
                    uint32_t dims[3] = { targetRows, targetCols, nIdx };
                    id<MTLBuffer> dimsBuf = [ctx->device newBufferWithBytes:dims
                                             length:sizeof(dims)
                                             options:MTLResourceStorageModeShared];

                    NSUInteger total = (NSUInteger)nIdx * (NSUInteger)targetCols;
                    if (!enc) enc = [cmdBuf computeCommandEncoder];
                    [enc setComputePipelineState:ctx->scatterAddRowsPSO];
                    [enc setBuffer:gBufferPool[targetId]  offset:0 atIndex:0];
                    [enc setBuffer:gBufferPool[indicesId] offset:0 atIndex:1];
                    [enc setBuffer:gBufferPool[gradId]    offset:0 atIndex:2];
                    [enc setBuffer:dimsBuf                offset:0 atIndex:3];
                    NSUInteger tpg = MIN(total, ctx->scatterAddRowsPSO.maxTotalThreadsPerThreadgroup);
                    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
                    pos += 7;
                    break;
                }

                // ── Scatter add rows (in-place atomic): [op, target, indices, grad, targetRows, targetCols, nIdx] ──
                case OP_SCATTER_ADD_ROWS_ATOMIC: {
                    int targetId = cmd[pos+1], indicesId = cmd[pos+2], gradId = cmd[pos+3];
                    uint32_t targetRows = (uint32_t)cmd[pos+4];
                    uint32_t targetCols = (uint32_t)cmd[pos+5];
                    uint32_t nIdx = (uint32_t)cmd[pos+6];
                    uint32_t dims[3] = { targetRows, targetCols, nIdx };
                    id<MTLBuffer> dimsBuf = [ctx->device newBufferWithBytes:dims
                                             length:sizeof(dims)
                                             options:MTLResourceStorageModeShared];

                    NSUInteger total = (NSUInteger)nIdx * (NSUInteger)targetCols;
                    if (!enc) enc = [cmdBuf computeCommandEncoder];
                    [enc setComputePipelineState:ctx->scatterAddRowsAtomicPSO];
                    [enc setBuffer:gBufferPool[targetId]  offset:0 atIndex:0];
                    [enc setBuffer:gBufferPool[indicesId] offset:0 atIndex:1];
                    [enc setBuffer:gBufferPool[gradId]    offset:0 atIndex:2];
                    [enc setBuffer:dimsBuf                offset:0 atIndex:3];
                    NSUInteger tpg = MIN(total, ctx->scatterAddRowsAtomicPSO.maxTotalThreadsPerThreadgroup);
                    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
                    pos += 7;
                    break;
                }

                // ── Row/col broadcasts: [op, a, vec, out, rows, cols] ──
                case OP_ADD_ROW_VECTOR:
                case OP_ADD_BROADCAST_COLS:
                case OP_SUBTRACT_BROADCAST_COLS:
                case OP_DIVIDE_BROADCAST_COLS:
                case OP_MULTIPLY_BROADCAST_ROWS:
                case OP_MULTIPLY_BROADCAST_COLS: {
                    int aId = cmd[pos+1], vecId = cmd[pos+2], outId = cmd[pos+3];
                    uint32_t rows = (uint32_t)cmd[pos+4];
                    uint32_t cols = (uint32_t)cmd[pos+5];
                    uint32_t dims[2] = { rows, cols };
                    id<MTLBuffer> dimsBuf = [ctx->device newBufferWithBytes:dims
                                             length:sizeof(dims)
                                             options:MTLResourceStorageModeShared];

                    id<MTLComputePipelineState> pso;
                    switch (opCode) {
                        case OP_ADD_ROW_VECTOR: pso = ctx->addRowVectorPSO; break;
                        case OP_ADD_BROADCAST_COLS: pso = ctx->addBroadcastColsPSO; break;
                        case OP_SUBTRACT_BROADCAST_COLS: pso = ctx->subtractBroadcastColsPSO; break;
                        case OP_DIVIDE_BROADCAST_COLS: pso = ctx->divideBroadcastColsPSO; break;
                        case OP_MULTIPLY_BROADCAST_ROWS: pso = ctx->multiplyBroadcastRowsPSO; break;
                        case OP_MULTIPLY_BROADCAST_COLS: pso = ctx->multiplyBroadcastColsPSO; break;
                        default: pso = ctx->addRowVectorPSO; break;
                    }

                    NSUInteger total = (NSUInteger)rows * (NSUInteger)cols;
                    if (!enc) enc = [cmdBuf computeCommandEncoder];
                    [enc setComputePipelineState:pso];
                    [enc setBuffer:gBufferPool[aId]   offset:0 atIndex:0];
                    [enc setBuffer:gBufferPool[vecId] offset:0 atIndex:1];
                    [enc setBuffer:gBufferPool[outId] offset:0 atIndex:2];
                    [enc setBuffer:dimsBuf            offset:0 atIndex:3];
                    NSUInteger tpg = MIN(total, pso.maxTotalThreadsPerThreadgroup);
                    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
                    pos += 6;
                    break;
                }

                // ── Reductions: [op, in, out, rows, cols] ───────────
                case OP_SUM_ROWS:
                case OP_SUM_ALONG_ROWS:
                case OP_MEAN_ALONG_ROWS:
                case OP_VARIANCE_ALONG_ROWS:
                case OP_MAX_ALONG_ROWS: {
                    int inId = cmd[pos+1], outId = cmd[pos+2];
                    uint32_t rows = (uint32_t)cmd[pos+3];
                    uint32_t cols = (uint32_t)cmd[pos+4];
                    uint32_t dims[2] = { rows, cols };
                    id<MTLBuffer> dimsBuf = [ctx->device newBufferWithBytes:dims
                                             length:sizeof(dims)
                                             options:MTLResourceStorageModeShared];

                    id<MTLComputePipelineState> pso;
                    bool rowWise;
                    switch (opCode) {
                        case OP_SUM_ROWS:
                            pso = ctx->sumRowsPSO;
                            rowWise = false;
                            break;
                        case OP_MEAN_ALONG_ROWS:
                            pso = ctx->meanAlongRowsPSO;
                            rowWise = true;
                            break;
                        case OP_SUM_ALONG_ROWS:
                            pso = ctx->sumAlongRowsPSO;
                            rowWise = true;
                            break;
                        case OP_VARIANCE_ALONG_ROWS:
                            pso = ctx->varianceAlongRowsPSO;
                            rowWise = true;
                            break;
                        case OP_MAX_ALONG_ROWS:
                            pso = ctx->maxAlongRowsPSO;
                            rowWise = true;
                            break;
                        default:
                            pso = ctx->sumRowsPSO;
                            rowWise = false;
                            break;
                    }

                    if (!enc) enc = [cmdBuf computeCommandEncoder];
                    [enc setComputePipelineState:pso];
                    [enc setBuffer:gBufferPool[inId]  offset:0 atIndex:0];
                    [enc setBuffer:gBufferPool[outId] offset:0 atIndex:1];
                    [enc setBuffer:dimsBuf            offset:0 atIndex:2];
                    if (rowWise) {
                        NSUInteger tpg = rowReductionWidth(pso);
                        [enc dispatchThreads:MTLSizeMake(tpg, (NSUInteger)rows, 1)
                            threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
                    } else {
                        NSUInteger tpg = colReductionHeight(pso);
                        [enc dispatchThreads:MTLSizeMake((NSUInteger)cols, tpg, 1)
                            threadsPerThreadgroup:MTLSizeMake(1, tpg, 1)];
                    }
                    pos += 5;
                    break;
                }

                // ── Scalar sum-abs reduction: [op, in, out, rows, cols] ──────
                case OP_SUM_ABS: {
                    int inId = cmd[pos+1], outId = cmd[pos+2];
                    uint32_t rows = (uint32_t)cmd[pos+3];
                    uint32_t cols = (uint32_t)cmd[pos+4];
                    uint32_t dims[2] = { rows, cols };
                    id<MTLBuffer> dimsBuf = [ctx->device newBufferWithBytes:dims
                                             length:sizeof(dims)
                                             options:MTLResourceStorageModeShared];

                    if (!enc) enc = [cmdBuf computeCommandEncoder];
                    [enc setComputePipelineState:ctx->sumAbsPSO];
                    [enc setBuffer:gBufferPool[inId]  offset:0 atIndex:0];
                    [enc setBuffer:gBufferPool[outId] offset:0 atIndex:1];
                    [enc setBuffer:dimsBuf            offset:0 atIndex:2];
                    NSUInteger tpg = MIN((NSUInteger)256, ctx->sumAbsPSO.maxTotalThreadsPerThreadgroup);
                    [enc dispatchThreads:MTLSizeMake(tpg, (NSUInteger)rows, 1)
                        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
                    pos += 5;
                    break;
                }

                // ── Scalar sum reduction: [op, in, out, rows, cols] ──────────
                case OP_SUM_SCALAR: {
                    int inId = cmd[pos+1], outId = cmd[pos+2];
                    uint32_t rows = (uint32_t)cmd[pos+3];
                    uint32_t cols = (uint32_t)cmd[pos+4];

                    id<MTLBuffer> tmpColSums = [ctx->device newBufferWithLength:(NSUInteger)cols * sizeof(float)
                                                                        options:MTLResourceStorageModeShared];

                    uint32_t dimsRows[2] = { rows, cols };
                    id<MTLBuffer> dimsRowsBuf = [ctx->device newBufferWithBytes:dimsRows
                                                 length:sizeof(dimsRows)
                                                 options:MTLResourceStorageModeShared];

                    uint32_t dimsScalar[2] = { 1u, cols };
                    id<MTLBuffer> dimsScalarBuf = [ctx->device newBufferWithBytes:dimsScalar
                                                   length:sizeof(dimsScalar)
                                                   options:MTLResourceStorageModeShared];

                    if (!enc) enc = [cmdBuf computeCommandEncoder];

                    // Pass 1: [rows x cols] -> [1 x cols]
                    [enc setComputePipelineState:ctx->sumRowsPSO];
                    [enc setBuffer:gBufferPool[inId] offset:0 atIndex:0];
                    [enc setBuffer:tmpColSums        offset:0 atIndex:1];
                    [enc setBuffer:dimsRowsBuf       offset:0 atIndex:2];
                    NSUInteger tpg1 = colReductionHeight(ctx->sumRowsPSO);
                    [enc dispatchThreads:MTLSizeMake((NSUInteger)cols, tpg1, 1)
                        threadsPerThreadgroup:MTLSizeMake(1, tpg1, 1)];

                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // Pass 2: [1 x cols] -> [1 x 1]
                    [enc setComputePipelineState:ctx->sumAlongRowsPSO];
                    [enc setBuffer:tmpColSums        offset:0 atIndex:0];
                    [enc setBuffer:gBufferPool[outId] offset:0 atIndex:1];
                    [enc setBuffer:dimsScalarBuf      offset:0 atIndex:2];
                    NSUInteger tpg2 = MIN((NSUInteger)1, ctx->sumAlongRowsPSO.maxTotalThreadsPerThreadgroup);
                    [enc dispatchThreads:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(tpg2, 1, 1)];

                    pos += 5;
                    break;
                }

                // ── Cross-entropy loss: [op, logits, targets, out, rows, cols] ─
                case OP_CROSS_ENTROPY_LOSS: {
                    int logitsId = cmd[pos+1], targetsId = cmd[pos+2], outId = cmd[pos+3];
                    uint32_t rows = (uint32_t)cmd[pos+4];
                    uint32_t cols = (uint32_t)cmd[pos+5];
                    uint32_t dims[2] = { rows, cols };
                    id<MTLBuffer> dimsBuf = [ctx->device newBufferWithBytes:dims
                                             length:sizeof(dims)
                                             options:MTLResourceStorageModeShared];

                    if (!enc) enc = [cmdBuf computeCommandEncoder];
                    [enc setComputePipelineState:ctx->crossEntropyLossPSO];
                    [enc setBuffer:gBufferPool[logitsId]  offset:0 atIndex:0];
                    [enc setBuffer:gBufferPool[targetsId] offset:0 atIndex:1];
                    [enc setBuffer:gBufferPool[outId]     offset:0 atIndex:2];
                    [enc setBuffer:dimsBuf                offset:0 atIndex:3];
                    NSUInteger tpg = MIN((NSUInteger)256, ctx->crossEntropyLossPSO.maxTotalThreadsPerThreadgroup);
                    [enc dispatchThreads:MTLSizeMake(tpg, (NSUInteger)rows, 1)
                        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
                    pos += 6;
                    break;
                }

                // ── Cross-entropy gradient: [op, logits, targets, out, rows, cols] ─
                case OP_CROSS_ENTROPY_GRADIENT: {
                    int logitsId = cmd[pos+1], targetsId = cmd[pos+2], outId = cmd[pos+3];
                    uint32_t rows = (uint32_t)cmd[pos+4];
                    uint32_t cols = (uint32_t)cmd[pos+5];
                    uint32_t dims[2] = { rows, cols };
                    id<MTLBuffer> dimsBuf = [ctx->device newBufferWithBytes:dims
                                             length:sizeof(dims)
                                             options:MTLResourceStorageModeShared];

                    if (!enc) enc = [cmdBuf computeCommandEncoder];
                    [enc setComputePipelineState:ctx->crossEntropyGradPSO];
                    [enc setBuffer:gBufferPool[logitsId]  offset:0 atIndex:0];
                    [enc setBuffer:gBufferPool[targetsId] offset:0 atIndex:1];
                    [enc setBuffer:gBufferPool[outId]     offset:0 atIndex:2];
                    [enc setBuffer:dimsBuf                offset:0 atIndex:3];
                    NSUInteger tpg = MIN((NSUInteger)256, ctx->crossEntropyGradPSO.maxTotalThreadsPerThreadgroup);
                    [enc dispatchThreads:MTLSizeMake(tpg, (NSUInteger)rows, 1)
                        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
                    pos += 6;
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
                    NSUInteger tpg = rowReductionWidth(ctx->softmaxBackwardPSO);
                    [enc dispatchThreads:MTLSizeMake(tpg, (NSUInteger)rows, 1)
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
                    NSUInteger tpg = rowReductionWidth(ctx->layerNormBackwardPSO);
                    [enc dispatchThreads:MTLSizeMake(tpg, (NSUInteger)rows, 1)
                        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
                    pos += 7;
                    break;
                }

                // ── AdamW update (in-place): [op, w, g, mt, vt, lr, b1, b2, eps, wd, bc1, bc2, n] ──
                case OP_ADAMW_UPDATE: {
                    int wId = cmd[pos+1], gId = cmd[pos+2], mtId = cmd[pos+3], vtId = cmd[pos+4];

                    id<MTLBuffer> wBuf = requireBuffer(wId, "OP_ADAMW_UPDATE");
                    id<MTLBuffer> gBuf = requireBuffer(gId, "OP_ADAMW_UPDATE");
                    id<MTLBuffer> mtBuf = requireBuffer(mtId, "OP_ADAMW_UPDATE");
                    id<MTLBuffer> vtBuf = requireBuffer(vtId, "OP_ADAMW_UPDATE");

                    auto bitsToFloat = [](int bits) {
                        float v;
                        std::memcpy(&v, &bits, sizeof(float));
                        return v;
                    };

                    struct AdamWParamsHost {
                        float lr;
                        float beta1;
                        float beta2;
                        float eps;
                        float weightDecay;
                        float bc1;
                        float bc2;
                    } params;

                    params.lr = bitsToFloat(cmd[pos+5]);
                    params.beta1 = bitsToFloat(cmd[pos+6]);
                    params.beta2 = bitsToFloat(cmd[pos+7]);
                    params.eps = bitsToFloat(cmd[pos+8]);
                    params.weightDecay = bitsToFloat(cmd[pos+9]);
                    params.bc1 = bitsToFloat(cmd[pos+10]);
                    params.bc2 = bitsToFloat(cmd[pos+11]);
                    int n = cmd[pos+12];

                    id<MTLBuffer> paramsBuf = [ctx->device newBufferWithBytes:&params
                                              length:sizeof(AdamWParamsHost)
                                              options:MTLResourceStorageModeShared];

                    if (!enc) enc = [cmdBuf computeCommandEncoder];
                    [enc setComputePipelineState:ctx->adamWUpdatePSO];
                    [enc setBuffer:wBuf              offset:0 atIndex:0];
                    [enc setBuffer:gBuf              offset:0 atIndex:1];
                    [enc setBuffer:mtBuf             offset:0 atIndex:2];
                    [enc setBuffer:vtBuf             offset:0 atIndex:3];
                    [enc setBuffer:paramsBuf         offset:0 atIndex:4];
                    NSUInteger tpg = MIN((NSUInteger)n, ctx->adamWUpdatePSO.maxTotalThreadsPerThreadgroup);
                    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
                    pos += 13;
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
