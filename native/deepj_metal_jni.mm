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
    id<MTLComputePipelineState> splitHeadsPSO;
    id<MTLComputePipelineState> mergeHeadsPSO;
    id<MTLComputePipelineState> causalMaskPSO;
    id<MTLComputePipelineState> causalSoftmaxPSO;
    id<MTLComputePipelineState> rotaryPSO;
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
    id<MTLComputePipelineState> sumSquaresPSO;
    id<MTLComputePipelineState> sumSquaresScalarPSO;
    id<MTLComputePipelineState> crossEntropyLossPSO;
    id<MTLComputePipelineState> crossEntropyGradPSO;
    id<MTLComputePipelineState> crossEntropyFusedPSO;
    id<MTLComputePipelineState> clampPSO;
    id<MTLComputePipelineState> powPSO;
    id<MTLComputePipelineState> scatterAddRowsPSO;
    id<MTLComputePipelineState> scatterAddRowsAtomicPSO;
    id<MTLComputePipelineState> gatherRowsPSO;
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
    id<MTLComputePipelineState> rmsNormPSO;
    id<MTLComputePipelineState> rmsNormBackwardPSO;
    id<MTLComputePipelineState> swiGluPSO;
    id<MTLComputePipelineState> swiGluBackwardPSO;
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

kernel void kernel_split_heads(device const float* input [[buffer(0)]],
                               device float* output      [[buffer(1)]],
                               device const uint4* dims  [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
    uint seq = dims[0].x, heads = dims[0].y, headDim = dims[0].z;
    uint modelWidth = dims[0].w, total = seq * heads * headDim;
    if (id >= total) return;
    uint dim = id % headDim, entry = id / headDim;
    uint position = entry % seq, head = entry / seq;
    output[id] = input[position * modelWidth + head * headDim + dim];
}

kernel void kernel_merge_heads(device const float* input [[buffer(0)]],
                               device float* output      [[buffer(1)]],
                               device const uint4* dims  [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
    uint seq = dims[0].x, heads = dims[0].y, headDim = dims[0].z;
    uint modelWidth = dims[0].w, total = seq * modelWidth;
    if (id >= total) return;
    uint position = id / modelWidth, column = id % modelWidth;
    uint head = column / headDim, dim = column % headDim;
    output[id] = input[(head * seq + position) * headDim + dim];
}

kernel void kernel_causal_mask(device const float* input [[buffer(0)]],
                               device float* output      [[buffer(1)]],
                               device const uint2* dims  [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
    uint rows = dims[0].x, seq = dims[0].y, total = rows * seq;
    if (id >= total) return;
    uint row = id / seq, column = id % seq, position = row % seq;
    output[id] = column > position ? -1.0e9f : input[id];
}

kernel void kernel_rotary(device const float* input  [[buffer(0)]],
                          device const float* cosine [[buffer(1)]],
                          device const float* sine   [[buffer(2)]],
                          device float* output       [[buffer(3)]],
                          device const uint4* dims   [[buffer(4)]],
                          uint id [[thread_position_in_grid]]) {
    uint rows = dims[0].x, seq = dims[0].y, headDim = dims[0].z;
    uint halfDim = headDim / 2, total = rows * halfDim;
    if (id >= total) return;
    uint row = id / halfDim, pair = id % halfDim, position = row % seq;
    uint inputIndex = row * headDim + pair * 2, tableIndex = position * halfDim + pair;
    float direction = dims[0].w == 0 ? 1.0f : -1.0f;
    float x = input[inputIndex], y = input[inputIndex + 1];
    float c = cosine[tableIndex], s = sine[tableIndex] * direction;
    output[inputIndex] = x * c - y * s;
    output[inputIndex + 1] = x * s + y * c;
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

inline float row_sum(device const float* values, uint base, uint cols,
                     uint tid, uint width) {
    float result = 0.0f;
    for (uint c = tid; c < cols; c += width) result += values[base + c];
    return result;
}

inline float row_abs_sum(device const float* values, uint base, uint cols,
                         uint tid, uint width) {
    float result = 0.0f;
    for (uint c = tid; c < cols; c += width) result += fabs(values[base + c]);
    return result;
}

inline float row_square_sum(device const float* values, uint base, uint cols,
                            uint tid, uint width) {
    float result = 0.0f;
    for (uint c = tid; c < cols; c += width) {
        float value = values[base + c];
        result += value * value;
    }
    return result;
}

inline float row_max(device const float* values, uint base, uint cols,
                     uint tid, uint width) {
    float result = -INFINITY;
    for (uint c = tid; c < cols; c += width) result = max(result, values[base + c]);
    return result;
}

inline float row_exp_sum(device const float* values, uint base, uint cols,
                         uint tid, uint width, float maximum) {
    float result = 0.0f;
    for (uint c = tid; c < cols; c += width) result += exp(values[base + c] - maximum);
    return result;
}

inline float row_variance_sum(device const float* values, uint base, uint cols,
                              uint tid, uint width, float mean) {
    float result = 0.0f;
    for (uint c = tid; c < cols; c += width) {
        float difference = values[base + c] - mean;
        result += difference * difference;
    }
    return result;
}

inline float column_sum(device const float* values, uint rows, uint cols,
                        uint column, uint tid, uint width) {
    float result = 0.0f;
    for (uint row = tid; row < rows; row += width) result += values[row * cols + column];
    return result;
}

inline float reduce_sum(threadgroup float* scratch, float value,
                        uint tid, uint width) {
    scratch[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = width >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return scratch[0];
}

inline float reduce_max(threadgroup float* scratch, float value,
                        uint tid, uint width) {
    scratch[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = width >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return scratch[0];
}

inline void write_row_value(device float* output, uint base, uint cols,
                            uint tid, uint width, float value) {
    for (uint c = tid; c < cols; c += width) output[base + c] = value;
}

inline bool valid_cross_entropy_target(device float* output, uint base, uint cols,
                                       uint tid, uint width, int target) {
    if (target >= 0 && target < (int)cols) return true;
    write_row_value(output, base, cols, tid, width, NAN);
    return false;
}

inline bool valid_cross_entropy_outputs(device float* losses, device float* gradient,
                                        uint row, uint base, uint cols,
                                        uint tid, uint width, int target) {
    if (valid_cross_entropy_target(gradient, base, cols, tid, width, target)) return true;
    if (tid == 0) losses[row] = NAN;
    return false;
}

inline float cross_entropy_value(device const float* logits, uint base, uint cols,
                                 int target, float maximum, float sumExp) {
    if (target < 0 || target >= (int)cols) return NAN;
    return log(sumExp) + maximum - logits[base + (uint)target];
}

inline void write_cross_entropy_gradient(device const float* logits, device float* output,
                                         uint base, uint cols, uint tid, uint width,
                                         int target, float maximum, float sumExp, float scale) {
    for (uint c = tid; c < cols; c += width) {
        float probability = exp(logits[base + c] - maximum) / sumExp;
        if ((int)c == target) probability -= 1.0f;
        output[base + c] = probability * scale;
    }
}

inline void write_rms_norm(device const float* input, device const float* gamma,
                           device float* output, device float* normalized,
                           uint base, uint cols, uint tid, uint width, float scale) {
    for (uint column = tid; column < cols; column += width) {
        float value = input[base + column] * scale;
        normalized[base + column] = value;
        output[base + column] = value * gamma[column];
    }
}

inline float rms_inner_sum(device const float* gradient, device const float* normalized,
                           device const float* gamma, uint base,
                           uint cols, uint tid, uint width) {
    float sum = 0.0f;
    for (uint column = tid; column < cols; column += width) {
        sum += gradient[base + column] * gamma[column] * normalized[base + column];
    }
    return sum;
}

inline void write_rms_gradient(device const float* gradient, device const float* normalized,
                               device const float* rms, device const float* gamma,
                               device float* output, uint row, uint base,
                               uint cols, uint tid, uint width, float inner) {
    for (uint column = tid; column < cols; column += width) {
        float scaled = gradient[base + column] * gamma[column];
        output[base + column] = (scaled - normalized[base + column] * inner) / rms[row];
    }
}

kernel void kernel_sum_rows(device const float* a      [[buffer(0)]],
                            device float* out          [[buffer(1)]],
                            device const uint2* dims   [[buffer(2)]],
                            uint3 gid [[thread_position_in_grid]],
                            uint3 tid3 [[thread_position_in_threadgroup]],
                            uint3 tptg [[threads_per_threadgroup]]) {
    uint rows = dims[0].x, cols = dims[0].y, col = gid.x, tid = tid3.y;
    if (col >= cols) return;
    threadgroup float scratch[1024];
    float value = column_sum(a, rows, cols, col, tid, tptg.y);
    float sum = reduce_sum(scratch, value, tid, tptg.y);
    if (tid == 0) out[col] = sum;
}

kernel void kernel_mean_along_rows(device const float* a      [[buffer(0)]],
                                   device float* out          [[buffer(1)]],
                                   device const uint2* dims   [[buffer(2)]],
                                   uint3 gid [[thread_position_in_grid]],
                                   uint3 tid3 [[thread_position_in_threadgroup]],
                                   uint3 tptg [[threads_per_threadgroup]]) {
    uint rows = dims[0].x, cols = dims[0].y, row = gid.y, tid = tid3.x;
    if (row >= rows) return;
    threadgroup float scratch[1024];
    float value = row_sum(a, row * cols, cols, tid, tptg.x);
    float sum = reduce_sum(scratch, value, tid, tptg.x);
    if (tid == 0) out[row] = sum / (float)cols;
}

kernel void kernel_sum_along_rows(device const float* a      [[buffer(0)]],
                                  device float* out          [[buffer(1)]],
                                  device const uint2* dims   [[buffer(2)]],
                                  uint3 gid [[thread_position_in_grid]],
                                  uint3 tid3 [[thread_position_in_threadgroup]],
                                  uint3 tptg [[threads_per_threadgroup]]) {
    uint rows = dims[0].x, cols = dims[0].y, row = gid.y, tid = tid3.x;
    if (row >= rows) return;
    threadgroup float scratch[1024];
    float value = row_sum(a, row * cols, cols, tid, tptg.x);
    float sum = reduce_sum(scratch, value, tid, tptg.x);
    if (tid == 0) out[row] = sum;
}

kernel void kernel_variance_along_rows(device const float* a      [[buffer(0)]],
                                       device float* out          [[buffer(1)]],
                                       device const uint2* dims   [[buffer(2)]],
                                       uint3 gid [[thread_position_in_grid]],
                                       uint3 tid3 [[thread_position_in_threadgroup]],
                                       uint3 tptg [[threads_per_threadgroup]]) {
    uint rows = dims[0].x, cols = dims[0].y, row = gid.y, tid = tid3.x;
    if (row >= rows) return;
    threadgroup float scratch[1024];
    uint base = row * cols;
    float mean = reduce_sum(scratch, row_sum(a, base, cols, tid, tptg.x),
                            tid, tptg.x) / (float)cols;
    float value = row_variance_sum(a, base, cols, tid, tptg.x, mean);
    float variance = reduce_sum(scratch, value, tid, tptg.x) / (float)cols;
    if (tid == 0) out[row] = variance;
}

kernel void kernel_max_along_rows(device const float* a      [[buffer(0)]],
                                  device float* out          [[buffer(1)]],
                                  device const uint2* dims   [[buffer(2)]],
                                  uint3 gid [[thread_position_in_grid]],
                                  uint3 tid3 [[thread_position_in_threadgroup]],
                                  uint3 tptg [[threads_per_threadgroup]]) {
    uint rows = dims[0].x, cols = dims[0].y, row = gid.y, tid = tid3.x;
    if (row >= rows) return;
    threadgroup float scratch[1024];
    float value = row_max(a, row * cols, cols, tid, tptg.x);
    float maximum = reduce_max(scratch, value, tid, tptg.x);
    if (tid == 0) out[row] = maximum;
}

kernel void kernel_sum_abs(device const float* a      [[buffer(0)]],
                           device float* out          [[buffer(1)]],
                           device const uint2* dims   [[buffer(2)]],
                           uint3 gid [[thread_position_in_grid]],
                           uint3 tid3 [[thread_position_in_threadgroup]],
                           uint3 tptg [[threads_per_threadgroup]]) {
    uint rows = dims[0].x, cols = dims[0].y, row = gid.y, tid = tid3.x;
    if (row >= rows) return;
    threadgroup float scratch[1024];
    float value = row_abs_sum(a, row * cols, cols, tid, tptg.x);
    float sum = reduce_sum(scratch, value, tid, tptg.x);
    if (tid == 0) out[row] = sum;
}

kernel void kernel_sum_squares(device const float* a    [[buffer(0)]],
                               device float* out        [[buffer(1)]],
                               device const uint2* dims [[buffer(2)]],
                               uint3 gid [[thread_position_in_grid]],
                               uint3 tid3 [[thread_position_in_threadgroup]],
                               uint3 tptg [[threads_per_threadgroup]]) {
    uint rows = dims[0].x, cols = dims[0].y, row = gid.y, tid = tid3.x;
    if (row >= rows) return;
    threadgroup float scratch[1024];
    float value = row_square_sum(a, row * cols, cols, tid, tptg.x);
    float sum = reduce_sum(scratch, value, tid, tptg.x);
    if (tid == 0) out[row] = sum;
}

kernel void kernel_cross_entropy_loss(device const float* logits [[buffer(0)]],
                                      device const float* targets [[buffer(1)]],
                                      device float* out           [[buffer(2)]],
                                      device const uint2* dims    [[buffer(3)]],
                                      uint3 gid [[thread_position_in_grid]],
                                      uint3 tid3 [[thread_position_in_threadgroup]],
                                      uint3 tptg [[threads_per_threadgroup]]) {
    uint rows = dims[0].x, cols = dims[0].y, row = gid.y, tid = tid3.x;
    if (row >= rows) return;
    threadgroup float scratch[1024];
    uint base = row * cols;
    float maximum = reduce_max(scratch, row_max(logits, base, cols, tid, tptg.x), tid, tptg.x);
    float localSum = row_exp_sum(logits, base, cols, tid, tptg.x, maximum);
    float sumExp = reduce_sum(scratch, localSum, tid, tptg.x);
    int target = (int)targets[row];
    if (tid == 0) out[row] = cross_entropy_value(logits, base, cols, target, maximum, sumExp);
}

kernel void kernel_cross_entropy_gradient(device const float* logits [[buffer(0)]],
                                          device const float* targets [[buffer(1)]],
                                          device float* out           [[buffer(2)]],
                                          device const uint2* dims    [[buffer(3)]],
                                          uint3 gid [[thread_position_in_grid]],
                                          uint3 tid3 [[thread_position_in_threadgroup]],
                                          uint3 tptg [[threads_per_threadgroup]]) {
    uint rows = dims[0].x, cols = dims[0].y, row = gid.y, tid = tid3.x;
    if (row >= rows) return;
    uint base = row * cols;
    int target = (int)targets[row];
    if (!valid_cross_entropy_target(out, base, cols, tid, tptg.x, target)) return;
    threadgroup float scratch[1024];
    float maximum = reduce_max(scratch, row_max(logits, base, cols, tid, tptg.x), tid, tptg.x);
    float localSum = row_exp_sum(logits, base, cols, tid, tptg.x, maximum);
    float sumExp = reduce_sum(scratch, localSum, tid, tptg.x);
    write_cross_entropy_gradient(logits, out, base, cols, tid, tptg.x,
                                 target, maximum, sumExp, 1.0f / (float)rows);
}

kernel void kernel_cross_entropy_fused(device const float* logits [[buffer(0)]],
                                       device const float* targets [[buffer(1)]],
                                       device float* losses        [[buffer(2)]],
                                       device float* gradient      [[buffer(3)]],
                                       device const uint2* dims    [[buffer(4)]],
                                       uint3 gid [[thread_position_in_grid]],
                                       uint3 tid3 [[thread_position_in_threadgroup]],
                                       uint3 tptg [[threads_per_threadgroup]]) {
    uint rows = dims[0].x, cols = dims[0].y, row = gid.y, tid = tid3.x;
    if (row >= rows) return;
    threadgroup float scratch[1024];
    uint base = row * cols;
    int target = (int)targets[row];
    if (!valid_cross_entropy_outputs(losses, gradient, row, base, cols, tid, tptg.x, target)) return;
    float maximum = reduce_max(scratch, row_max(logits, base, cols, tid, tptg.x), tid, tptg.x);
    float sumExp = reduce_sum(scratch, row_exp_sum(logits, base, cols, tid, tptg.x, maximum), tid, tptg.x);
    if (tid == 0) losses[row] = cross_entropy_value(logits, base, cols, target, maximum, sumExp);
    write_cross_entropy_gradient(logits, gradient, base, cols, tid, tptg.x,
                                 target, maximum, sumExp, 1.0f / (float)rows);
}

struct RmsNormParams {
    uint rows;
    uint cols;
    float epsilon;
};

kernel void kernel_rms_norm(device const float* input [[buffer(0)]],
                            device const float* gamma [[buffer(1)]],
                            device float* output [[buffer(2)]],
                            device float* normalized [[buffer(3)]],
                            device float* rms [[buffer(4)]],
                            device const RmsNormParams* params [[buffer(5)]],
                            uint3 gid [[thread_position_in_grid]],
                            uint3 tid3 [[thread_position_in_threadgroup]],
                            uint3 tptg [[threads_per_threadgroup]]) {
    uint row = gid.y, tid = tid3.x, cols = params->cols;
    if (row >= params->rows) return;
    threadgroup float scratch[1024];
    uint base = row * cols;
    float squares = reduce_sum(scratch, row_square_sum(input, base, cols, tid, tptg.x), tid, tptg.x);
    float rmsValue = sqrt(squares / (float)cols + params->epsilon);
    if (tid == 0) rms[row] = rmsValue;
    write_rms_norm(input, gamma, output, normalized, base, cols, tid, tptg.x, 1.0f / rmsValue);
}

kernel void kernel_rms_norm_backward(device const float* gradient [[buffer(0)]],
                                     device const float* normalized [[buffer(1)]],
                                     device const float* rms [[buffer(2)]],
                                     device const float* gamma [[buffer(3)]],
                                     device float* output [[buffer(4)]],
                                     device const uint2* dims [[buffer(5)]],
                                     uint3 gid [[thread_position_in_grid]],
                                     uint3 tid3 [[thread_position_in_threadgroup]],
                                     uint3 tptg [[threads_per_threadgroup]]) {
    uint row = gid.y, tid = tid3.x, cols = dims[0].y;
    if (row >= dims[0].x) return;
    threadgroup float scratch[1024];
    uint base = row * cols;
    float local = rms_inner_sum(gradient, normalized, gamma, base, cols, tid, tptg.x);
    float inner = reduce_sum(scratch, local, tid, tptg.x) / (float)cols;
    write_rms_gradient(gradient, normalized, rms, gamma, output,
                       row, base, cols, tid, tptg.x, inner);
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

kernel void kernel_gather_rows(device const float* input   [[buffer(0)]],
                               device const float* indices [[buffer(1)]],
                               device float* output        [[buffer(2)]],
                               device const uint3* dims    [[buffer(3)]],
                               uint id [[thread_position_in_grid]]) {
    uint inputRows = dims[0].x, cols = dims[0].y, outputRows = dims[0].z;
    uint total = outputRows * cols;
    if (id >= total) return;
    uint outputRow = id / cols, column = id % cols;
    uint inputRow = (uint)indices[outputRow];
    output[id] = inputRow < inputRows ? input[inputRow * cols + column] : 0.0f;
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

kernel void kernel_sum_squares_scalar(device const float* input [[buffer(0)]],
                                      device atomic_uint* total [[buffer(1)]],
                                      device const uint2* dims [[buffer(2)]],
                                      uint id [[thread_position_in_grid]],
                                      uint tid [[thread_position_in_threadgroup]],
                                      uint width [[threads_per_threadgroup]]) {
    uint count = dims[0].x, gridWidth = dims[0].y * width;
    threadgroup float scratch[1024];
    float value = 0.0f;
    for (uint index = id; index < count; index += gridWidth) {
        value += input[index] * input[index];
    }
    float sum = reduce_sum(scratch, value, tid, width);
    if (tid == 0) atomic_add_f32(total, sum);
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

kernel void kernel_swiglu(device const float* gate [[buffer(0)]],
                          device const float* up [[buffer(1)]],
                          device float* output [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    float sigmoid = 1.0f / (1.0f + exp(-gate[id]));
    output[id] = gate[id] * sigmoid * up[id];
}

kernel void kernel_swiglu_backward(device const float* gradient [[buffer(0)]],
                                   device const float* gate [[buffer(1)]],
                                   device const float* up [[buffer(2)]],
                                   device float* gateGradient [[buffer(3)]],
                                   device float* upGradient [[buffer(4)]],
                                   uint id [[thread_position_in_grid]]) {
    float sigmoid = 1.0f / (1.0f + exp(-gate[id]));
    float activated = gate[id] * sigmoid;
    float derivative = sigmoid + activated * (1.0f - sigmoid);
    gateGradient[id] = gradient[id] * up[id] * derivative;
    upGradient[id] = gradient[id] * activated;
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

inline float write_row_exp(device const float* input, device float* output,
                           uint base, uint cols, uint tid, uint width, float maximum) {
    float sum = 0.0f;
    for (uint c = tid; c < cols; c += width) {
        float value = exp(input[base + c] - maximum);
        output[base + c] = value;
        sum += value;
    }
    return sum;
}

inline float row_dot(device const float* left, device const float* right,
                     uint base, uint cols, uint tid, uint width) {
    float result = 0.0f;
    for (uint c = tid; c < cols; c += width) result += left[base + c] * right[base + c];
    return result;
}

inline void write_softmax_backward(device const float* gradient, device const float* softmax,
                                   device float* output, uint base, uint cols,
                                   uint tid, uint width, float dot) {
    for (uint c = tid; c < cols; c += width) {
        float value = softmax[base + c];
        output[base + c] = value * (gradient[base + c] - dot);
    }
}

inline float2 layernorm_sums(device const float* gradient, device const float* normalized,
                             uint base, uint cols, uint tid, uint width) {
    float2 result = float2(0.0f);
    for (uint c = tid; c < cols; c += width) {
        float value = gradient[base + c];
        result.x += value;
        result.y += value * normalized[base + c];
    }
    return result;
}

inline float2 reduce_pair_sum(threadgroup float* left, threadgroup float* right,
                              float2 value, uint tid, uint width) {
    left[tid] = value.x;
    right[tid] = value.y;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = width >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            left[tid] += left[tid + stride];
            right[tid] += right[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return float2(left[0], right[0]);
}

inline void write_layernorm_backward(device const float* gradient,
                                     device const float* normalized, device float* output,
                                     uint base, uint cols, uint tid, uint width,
                                     float inverseDeviation, float2 sums) {
    float inverseColumns = 1.0f / (float)cols;
    for (uint c = tid; c < cols; c += width) {
        float value = gradient[base + c];
        float centered = value - sums.x * inverseColumns;
        output[base + c] = inverseDeviation *
                (centered - normalized[base + c] * sums.y * inverseColumns);
    }
}

kernel void kernel_softmax_max(device const float* a     [[buffer(0)]],
                               device float* rowMax      [[buffer(1)]],
                               device const uint* dims   [[buffer(2)]],
                               uint3 gid [[thread_position_in_grid]],
                               uint3 tid3 [[thread_position_in_threadgroup]],
                               uint3 tptg [[threads_per_threadgroup]]) {
    uint cols = dims[0], row = gid.y, tid = tid3.x;
    threadgroup float scratch[1024];
    uint base = row * cols;
    float maximum = reduce_max(scratch, row_max(a, base, cols, tid, tptg.x), tid, tptg.x);
    if (tid == 0) rowMax[row] = maximum;
}

struct CausalSoftmaxParams {
    uint rows;
    uint cols;
    uint sequenceLength;
    float scale;
};

inline float causal_max(device const float* input, uint base, uint position,
                        float scale, uint tid, uint width) {
    float result = -INFINITY;
    for (uint col = tid; col <= position; col += width) {
        result = max(result, input[base + col] * scale);
    }
    return result;
}

inline float causal_exp(device const float* input, device float* output,
                        uint base, uint cols, uint position, float scale,
                        float maximum, uint tid, uint width) {
    float result = 0.0f;
    for (uint col = tid; col < cols; col += width) {
        float value = col <= position ? exp(input[base + col] * scale - maximum) : 0.0f;
        output[base + col] = value;
        result += value;
    }
    return result;
}

inline void normalize_causal_row(device float* output, uint base, uint cols,
                                 float sum, uint tid, uint width) {
    for (uint col = tid; col < cols; col += width) output[base + col] /= sum;
}

kernel void kernel_causal_softmax(device const float* input [[buffer(0)]],
                                  device float* output      [[buffer(1)]],
                                  device const CausalSoftmaxParams* params [[buffer(2)]],
                                  uint3 gid [[thread_position_in_grid]],
                                  uint3 tid3 [[thread_position_in_threadgroup]],
                                  uint3 tptg [[threads_per_threadgroup]]) {
    uint row = gid.y, tid = tid3.x, cols = params[0].cols;
    if (row >= params[0].rows) return;
    uint base = row * cols, position = row % params[0].sequenceLength;
    threadgroup float scratch[1024];
    float localMax = causal_max(input, base, position, params[0].scale, tid, tptg.x);
    float maximum = reduce_max(scratch, localMax, tid, tptg.x);
    float localSum = causal_exp(input, output, base, cols, position,
                                params[0].scale, maximum, tid, tptg.x);
    float sum = reduce_sum(scratch, localSum, tid, tptg.x);
    normalize_causal_row(output, base, cols, sum, tid, tptg.x);
}

kernel void kernel_softmax_expsum(device const float* a     [[buffer(0)]],
                                  device float* out         [[buffer(1)]],
                                  device const float* rowMax[[buffer(2)]],
                                  device float* rowSum      [[buffer(3)]],
                                  device const uint* dims   [[buffer(4)]],
                                  uint3 gid [[thread_position_in_grid]],
                                  uint3 tid3 [[thread_position_in_threadgroup]],
                                  uint3 tptg [[threads_per_threadgroup]]) {
    uint cols = dims[0], row = gid.y, tid = tid3.x;
    threadgroup float scratch[1024];
    uint base = row * cols;
    float local = write_row_exp(a, out, base, cols, tid, tptg.x, rowMax[row]);
    float sum = reduce_sum(scratch, local, tid, tptg.x);
    if (tid == 0) rowSum[row] = sum;
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
    uint cols = dims[0], row = gid.y, tid = tid3.x;
    threadgroup float scratch[1024];
    uint base = row * cols;
    float local = row_dot(gradOutput, softmaxOut, base, cols, tid, tptg.x);
    float dot = reduce_sum(scratch, local, tid, tptg.x);
    write_softmax_backward(gradOutput, softmaxOut, out, base, cols, tid, tptg.x, dot);
}

kernel void kernel_layernorm_backward(device const float* dXHat [[buffer(0)]],
                                      device const float* xHat  [[buffer(1)]],
                                      device const float* std   [[buffer(2)]],
                                      device float* out         [[buffer(3)]],
                                      device const uint* dims   [[buffer(4)]],
                                      uint3 gid [[thread_position_in_grid]],
                                      uint3 tid3 [[thread_position_in_threadgroup]],
                                      uint3 tptg [[threads_per_threadgroup]]) {
    uint cols = dims[0], row = gid.y, tid = tid3.x;
    threadgroup float scratchA[1024];
    threadgroup float scratchB[1024];
    uint base = row * cols;
    float2 local = layernorm_sums(dXHat, xHat, base, cols, tid, tptg.x);
    float2 sums = reduce_pair_sum(scratchA, scratchB, local, tid, tptg.x);
    write_layernorm_backward(dXHat, xHat, out, base, cols, tid, tptg.x,
                             1.0f / std[row], sums);
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

static void initAttentionPipelines(MetalContext* ctx, id<MTLLibrary> library) {
    ctx->splitHeadsPSO = makePSO(library, @"kernel_split_heads");
    ctx->mergeHeadsPSO = makePSO(library, @"kernel_merge_heads");
    ctx->causalMaskPSO = makePSO(library, @"kernel_causal_mask");
    ctx->causalSoftmaxPSO = makePSO(library, @"kernel_causal_softmax");
    ctx->rotaryPSO = makePSO(library, @"kernel_rotary");
    ctx->gatherRowsPSO = makePSO(library, @"kernel_gather_rows");
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
    ctx->sumSquaresPSO = makePSO(library, @"kernel_sum_squares");
    ctx->sumSquaresScalarPSO = makePSO(library, @"kernel_sum_squares_scalar");
}

static void initLossPipelines(MetalContext* ctx, id<MTLLibrary> library) {
    ctx->crossEntropyLossPSO = makePSO(library, @"kernel_cross_entropy_loss");
    ctx->crossEntropyGradPSO = makePSO(library, @"kernel_cross_entropy_gradient");
    ctx->crossEntropyFusedPSO = makePSO(library, @"kernel_cross_entropy_fused");
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
    ctx->rmsNormPSO = makePSO(library, @"kernel_rms_norm");
    ctx->rmsNormBackwardPSO = makePSO(library, @"kernel_rms_norm_backward");
    ctx->swiGluPSO = makePSO(library, @"kernel_swiglu");
    ctx->swiGluBackwardPSO = makePSO(library, @"kernel_swiglu_backward");
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
    initAttentionPipelines(ctx, library);
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

using BufferPool = std::unordered_map<int, id<MTLBuffer>>;
using BufferEntry = BufferPool::iterator;

static BufferPool gBufferPool;
static std::mutex gBufferMutex;

static std::unordered_map<uint64_t, MPSMatrixMultiplication*> gMatmulKernelCaches[4];

static MPSMatrixMultiplication* cachedMatmulKernel(
        MetalContext* ctx, int m, int n, int k, bool transposeLeft, bool transposeRight) {
    uint64_t key = ((uint64_t)(uint32_t)m << 42) | ((uint64_t)(uint32_t)n << 21) | (uint64_t)(uint32_t)k;
    int flags = (transposeLeft ? 2 : 0) | (transposeRight ? 1 : 0);
    auto& cache = gMatmulKernelCaches[flags];
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    MPSMatrixMultiplication* mm =
        [[MPSMatrixMultiplication alloc] initWithDevice:ctx->device
            transposeLeft:transposeLeft transposeRight:transposeRight
            resultRows:m resultColumns:n interiorColumns:k
            alpha:1.0 beta:0.0];
    cache[key] = mm;
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
static constexpr int OP_SUM_SQUARES = 43;
static constexpr int OP_BATCHED_MATMUL = 44;
static constexpr int OP_SPLIT_HEADS = 45;
static constexpr int OP_MERGE_HEADS = 46;
static constexpr int OP_CAUSAL_MASK = 47;
static constexpr int OP_ROTARY = 48;
static constexpr int OP_GATHER_ROWS = 49;
static constexpr int OP_CAUSAL_SOFTMAX = 50;
static constexpr int OP_CROSS_ENTROPY_FUSED = 51;
static constexpr int OP_SUM_SQUARES_SCALAR = 52;
static constexpr int OP_RMS_NORM = 53;
static constexpr int OP_RMS_NORM_BACKWARD = 54;
static constexpr int OP_SWIGLU = 55;
static constexpr int OP_SWIGLU_BACKWARD = 56;

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

struct AllocationArrays {
    jint* ids;
    jint* sizes;
};

static AllocationArrays acquireAllocationArrays(JNIEnv* env, jintArray ids, jintArray sizes) {
    jint* idValues = env->GetIntArrayElements(ids, nullptr);
    jint* sizeValues = env->GetIntArrayElements(sizes, nullptr);
    if (idValues != nullptr && sizeValues != nullptr) {
        return AllocationArrays{idValues, sizeValues};
    }
    if (idValues != nullptr) env->ReleaseIntArrayElements(ids, idValues, JNI_ABORT);
    if (sizeValues != nullptr) env->ReleaseIntArrayElements(sizes, sizeValues, JNI_ABORT);
    return AllocationArrays{nullptr, nullptr};
}

static void releaseAllocationArrays(JNIEnv* env, jintArray ids, jintArray sizes,
                                    AllocationArrays arrays) {
    if (arrays.ids != nullptr) env->ReleaseIntArrayElements(ids, arrays.ids, JNI_ABORT);
    if (arrays.sizes != nullptr) env->ReleaseIntArrayElements(sizes, arrays.sizes, JNI_ABORT);
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

static BufferEntry transferEntry(JNIEnv* env, int bufferId, jfloatArray values,
                                 const char* operation) {
    if (values == nullptr) {
        throwJavaRuntimeException(env, (std::string(operation) + " data cannot be null").c_str());
        return gBufferPool.end();
    }
    auto entry = gBufferPool.find(bufferId);
    if (entry == gBufferPool.end()) {
        throwJavaRuntimeException(env, (std::string(operation) + " buffer was not found").c_str());
        return entry;
    }
    NSUInteger bytes = (NSUInteger)env->GetArrayLength(values) * sizeof(float);
    if (bytes == [entry->second length]) return entry;
    throwJavaRuntimeException(env, (std::string(operation) + " length does not match buffer size").c_str());
    return gBufferPool.end();
}

static void allocateRequestedBuffers(JNIEnv* env, jintArray idsArr,
                                     jintArray sizesArr, jint count) {
    if (!validateAllocationArrays(env, idsArr, sizesArr, count)) return;
    AllocationArrays arrays = acquireAllocationArrays(env, idsArr, sizesArr);
    if (arrays.ids == nullptr || arrays.sizes == nullptr) return;
    try {
        std::lock_guard<std::mutex> lock(gBufferMutex);
        allocateBuffers(arrays.ids, arrays.sizes, count);
    } catch (const std::exception& ex) {
        releaseAllocationArrays(env, idsArr, sizesArr, arrays);
        throwJavaRuntimeException(env, ex.what());
        return;
    }
    releaseAllocationArrays(env, idsArr, sizesArr, arrays);
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_nativeAllocBuffers(
        JNIEnv* env, jclass, jintArray idsArr, jintArray sizesArr, jint count) {
    @autoreleasepool { allocateRequestedBuffers(env, idsArr, sizesArr, count); }
}

static void uploadBuffer(JNIEnv* env, jint bufId, jfloatArray dataArr) {
    std::lock_guard<std::mutex> lock(gBufferMutex);
    BufferEntry entry = transferEntry(env, bufId, dataArr, "Upload");
    if (entry == gBufferPool.end()) return;
    jint len = env->GetArrayLength(dataArr);
    jfloat* data = env->GetFloatArrayElements(dataArr, nullptr);
    if (data == nullptr) return;
    std::memcpy([entry->second contents], data, (size_t)len * sizeof(float));
    env->ReleaseFloatArrayElements(dataArr, data, JNI_ABORT);
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_nativeUploadBuffer(
        JNIEnv* env, jclass, jint bufId, jfloatArray dataArr) {
    @autoreleasepool { uploadBuffer(env, bufId, dataArr); }
}

static void downloadBuffer(JNIEnv* env, jint bufId, jfloatArray outArr) {
    std::lock_guard<std::mutex> lock(gBufferMutex);
    BufferEntry entry = transferEntry(env, bufId, outArr, "Download");
    if (entry == gBufferPool.end()) return;
    jint len = env->GetArrayLength(outArr);
    jfloat* out = env->GetFloatArrayElements(outArr, nullptr);
    if (out == nullptr) return;
    std::memcpy(out, [entry->second contents], (size_t)len * sizeof(float));
    env->ReleaseFloatArrayElements(outArr, out, 0);
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_nativeDownloadBuffer(
        JNIEnv* env, jclass, jint bufId, jfloatArray outArr) {
    @autoreleasepool { downloadBuffer(env, bufId, outArr); }
}

static bool validateReleaseRequest(JNIEnv* env, jintArray idsArr, jint count) {
    if (idsArr != nullptr && count >= 0 && count <= env->GetArrayLength(idsArr)) return true;
    throwJavaRuntimeException(env, "Invalid GPU release request");
    return false;
}

static void releaseBuffers(JNIEnv* env, jintArray idsArr, jint count) {
    if (!validateReleaseRequest(env, idsArr, count)) return;
    std::lock_guard<std::mutex> lock(gBufferMutex);
    jint* ids = env->GetIntArrayElements(idsArr, nullptr);
    if (ids == nullptr) return;
    for (int i = 0; i < count; i++) gBufferPool.erase(ids[i]);
    env->ReleaseIntArrayElements(idsArr, ids, JNI_ABORT);
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_nativeReleaseBuffers(
        JNIEnv* env, jclass, jintArray idsArr, jint count) {
    @autoreleasepool { releaseBuffers(env, idsArr, count); }
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

static id<MTLComputePipelineState> headPermutationPipeline(MetalContext* ctx, int op) {
    if (op == OP_SPLIT_HEADS) return ctx->splitHeadsPSO;
    if (op == OP_MERGE_HEADS) return ctx->mergeHeadsPSO;
    throw std::runtime_error("Invalid head permutation op");
}

static void encodeHeadPermutation(GraphState& state, const jint* cmd, int pos) {
    uint32_t dims[] = {(uint32_t)cmd[pos + 3], (uint32_t)cmd[pos + 4],
                       (uint32_t)cmd[pos + 5], (uint32_t)cmd[pos + 6]};
    NSUInteger total = elementCount(cmd[pos + 3], cmd[pos + 6]);
    id<MTLBuffer> dimensions = valueBuffer(state.ctx, dims, sizeof(dims));
    id<MTLComputePipelineState> pipeline = headPermutationPipeline(state.ctx, cmd[pos]);
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "head permutation") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "head permutation") offset:0 atIndex:1];
    [encoder setBuffer:dimensions offset:0 atIndex:2];
    dispatch1D(encoder, pipeline, total);
}

static void encodeCausalMask(GraphState& state, const jint* cmd, int pos) {
    uint32_t dims[] = {(uint32_t)cmd[pos + 3], (uint32_t)cmd[pos + 4]};
    NSUInteger total = elementCount(cmd[pos + 3], cmd[pos + 4]);
    id<MTLBuffer> dimensions = valueBuffer(state.ctx, dims, sizeof(dims));
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->causalMaskPSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "causal mask") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "causal mask") offset:0 atIndex:1];
    [encoder setBuffer:dimensions offset:0 atIndex:2];
    dispatch1D(encoder, state.ctx->causalMaskPSO, total);
}

struct CausalSoftmaxParamsHost {
    uint32_t rows, cols, sequenceLength;
    float scale;
};

static void encodeCausalSoftmax(GraphState& state, const jint* cmd, int pos) {
    CausalSoftmaxParamsHost params{
        (uint32_t)positiveCount(cmd[pos + 3], "causal softmax rows"),
        (uint32_t)positiveCount(cmd[pos + 4], "causal softmax columns"),
        (uint32_t)positiveCount(cmd[pos + 5], "causal softmax sequence"),
        floatFromBits(cmd[pos + 6])};
    id<MTLBuffer> values = valueBuffer(state.ctx, &params, sizeof(params));
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->causalSoftmaxPSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "causal softmax") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "causal softmax") offset:0 atIndex:1];
    [encoder setBuffer:values offset:0 atIndex:2];
    NSUInteger threads = rowReductionWidth(state.ctx->causalSoftmaxPSO);
    [encoder dispatchThreads:MTLSizeMake(threads, params.rows, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static void encodeRotary(GraphState& state, const jint* cmd, int pos) {
    uint32_t dims[] = {(uint32_t)cmd[pos + 5], (uint32_t)cmd[pos + 6],
                       (uint32_t)cmd[pos + 7], (uint32_t)cmd[pos + 8]};
    NSUInteger total = elementCount(cmd[pos + 5], cmd[pos + 7] / 2);
    id<MTLBuffer> dimensions = valueBuffer(state.ctx, dims, sizeof(dims));
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->rotaryPSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "rotary input") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "rotary cosine") offset:0 atIndex:1];
    [encoder setBuffer:requireBuffer(cmd[pos + 3], "rotary sine") offset:0 atIndex:2];
    [encoder setBuffer:requireBuffer(cmd[pos + 4], "rotary output") offset:0 atIndex:3];
    [encoder setBuffer:dimensions offset:0 atIndex:4];
    dispatch1D(encoder, state.ctx->rotaryPSO, total);
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

static void encodeGatherRows(GraphState& state, const jint* cmd, int pos) {
    uint32_t dims[] = {(uint32_t)cmd[pos + 4], (uint32_t)cmd[pos + 5],
                       (uint32_t)cmd[pos + 6]};
    NSUInteger total = elementCount(cmd[pos + 5], cmd[pos + 6]);
    id<MTLBuffer> dimensions = valueBuffer(state.ctx, dims, sizeof(dims));
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->gatherRowsPSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "gather input") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "gather indices") offset:0 atIndex:1];
    [encoder setBuffer:requireBuffer(cmd[pos + 3], "gather output") offset:0 atIndex:2];
    [encoder setBuffer:dimensions offset:0 atIndex:3];
    dispatch1D(encoder, state.ctx->gatherRowsPSO, total);
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

static id<MTLComputePipelineState> rowStatisticPipeline(MetalContext* ctx, int op) {
    if (op == OP_SUM_ABS) return ctx->sumAbsPSO;
    if (op == OP_SUM_SQUARES) return ctx->sumSquaresPSO;
    throw std::runtime_error("Invalid row statistic op");
}

static void encodeRowStatistic(GraphState& state, const jint* cmd, int pos) {
    NSUInteger rows = positiveCount(cmd[pos + 3], "rows");
    uint32_t dims[] = {(uint32_t)rows, (uint32_t)positiveCount(cmd[pos + 4], "cols")};
    id<MTLBuffer> dimsBuffer = valueBuffer(state.ctx, dims, sizeof(dims));
    id<MTLComputePipelineState> pipeline = rowStatisticPipeline(state.ctx, cmd[pos]);
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "row statistic") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "row statistic") offset:0 atIndex:1];
    [encoder setBuffer:dimsBuffer offset:0 atIndex:2];
    NSUInteger threads = MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(threads, rows, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static void encodeSumSquaresScalar(GraphState& state, const jint* cmd, int pos) {
    NSUInteger count = positiveCount(cmd[pos + 3], "sum squares count");
    NSUInteger threads = MIN((NSUInteger)256,
                             state.ctx->sumSquaresScalarPSO.maxTotalThreadsPerThreadgroup);
    NSUInteger groups = MIN((count + threads - 1) / threads, (NSUInteger)1024);
    uint32_t dims[] = {(uint32_t)count, (uint32_t)groups};
    id<MTLBuffer> values = valueBuffer(state.ctx, dims, sizeof(dims));
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->sumSquaresScalarPSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "sum squares") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "sum squares") offset:0 atIndex:1];
    [encoder setBuffer:values offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
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

static void encodeFusedCrossEntropy(GraphState& state, const jint* cmd, int pos) {
    NSUInteger rows = positiveCount(cmd[pos + 5], "rows");
    uint32_t dims[] = {(uint32_t)rows, (uint32_t)positiveCount(cmd[pos + 6], "cols")};
    id<MTLBuffer> values = valueBuffer(state.ctx, dims, sizeof(dims));
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->crossEntropyFusedPSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "cross entropy") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "cross entropy") offset:0 atIndex:1];
    [encoder setBuffer:requireBuffer(cmd[pos + 3], "cross entropy") offset:0 atIndex:2];
    [encoder setBuffer:requireBuffer(cmd[pos + 4], "cross entropy") offset:0 atIndex:3];
    [encoder setBuffer:values offset:0 atIndex:4];
    NSUInteger threads = rowReductionWidth(state.ctx->crossEntropyFusedPSO);
    [encoder dispatchThreads:MTLSizeMake(threads, rows, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static MPSMatrixDescriptor* matrixDescriptor(int rows, int cols) {
    positiveCount(rows, "matrix rows");
    positiveCount(cols, "matrix cols");
    return [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols
        rowBytes:(NSUInteger)cols * sizeof(float) dataType:MPSDataTypeFloat32];
}

static MPSMatrixDescriptor* matrixBatchDescriptor(int rows, int cols, int batches) {
    positiveCount(rows, "matrix rows");
    positiveCount(cols, "matrix cols");
    positiveCount(batches, "matrix batches");
    NSUInteger rowBytes = (NSUInteger)cols * sizeof(float);
    NSUInteger matrixBytes = (NSUInteger)rows * rowBytes;
    return [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols
        matrices:batches rowBytes:rowBytes matrixBytes:matrixBytes
        dataType:MPSDataTypeFloat32];
}

static MPSMatrix* matrixBatch(id<MTLBuffer> buffer, int rows, int cols, int batches) {
    return [[MPSMatrix alloc] initWithBuffer:buffer
        descriptor:matrixBatchDescriptor(rows, cols, batches)];
}

static MPSMatrixMultiplication* configuredMatmulKernel(
        MetalContext* ctx, int m, int n, int k, bool transposeLeft,
        bool transposeRight, int batches) {
    MPSMatrixMultiplication* kernel = cachedMatmulKernel(
        ctx, m, n, k, transposeLeft, transposeRight);
    kernel.batchStart = 0;
    kernel.batchSize = batches;
    return kernel;
}

static void endGraphEncoder(GraphState& state) {
    if (state.encoder == nil) return;
    [state.encoder endEncoding];
    state.encoder = nil;
}

static void encodeMatmul(GraphState& state, const jint* cmd, int pos) {
    endGraphEncoder(state);
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
    [configuredMatmulKernel(state.ctx, m, n, k, false, false, 1)
        encodeToCommandBuffer:state.commandBuffer leftMatrix:a rightMatrix:b resultMatrix:out];
}

struct BatchedMatmulSpec {
    int batches, leftRows, leftCols, rightRows, rightCols;
    int rows, cols, inner;
    bool transposeLeft, transposeRight;
};

static BatchedMatmulSpec batchedMatmulSpec(const jint* cmd, int pos) {
    int batches = (int)positiveCount(cmd[pos + 4], "batches");
    int leftRows = (int)positiveCount(cmd[pos + 5], "left rows");
    int leftCols = (int)positiveCount(cmd[pos + 6], "left cols");
    int rightRows = (int)positiveCount(cmd[pos + 7], "right rows");
    int rightCols = (int)positiveCount(cmd[pos + 8], "right cols");
    bool transposeLeft = cmd[pos + 9] != 0, transposeRight = cmd[pos + 10] != 0;
    int rows = transposeLeft ? leftCols : leftRows;
    int inner = transposeLeft ? leftRows : leftCols;
    int rightInner = transposeRight ? rightCols : rightRows;
    int cols = transposeRight ? rightRows : rightCols;
    if (inner != rightInner) throw std::runtime_error("Batched matmul shape mismatch");
    return {batches, leftRows, leftCols, rightRows, rightCols,
            rows, cols, inner, transposeLeft, transposeRight};
}

static void encodeBatchedMatmul(GraphState& state, const jint* cmd, int pos) {
    BatchedMatmulSpec spec = batchedMatmulSpec(cmd, pos);
    endGraphEncoder(state);
    MPSMatrix* left = matrixBatch(requireBuffer(cmd[pos + 1], "batched matmul"),
                                  spec.leftRows, spec.leftCols, spec.batches);
    MPSMatrix* right = matrixBatch(requireBuffer(cmd[pos + 2], "batched matmul"),
                                   spec.rightRows, spec.rightCols, spec.batches);
    MPSMatrix* output = matrixBatch(requireBuffer(cmd[pos + 3], "batched matmul"),
                                    spec.rows, spec.cols, spec.batches);
    [configuredMatmulKernel(state.ctx, spec.rows, spec.cols, spec.inner,
                            spec.transposeLeft, spec.transposeRight, spec.batches)
        encodeToCommandBuffer:state.commandBuffer leftMatrix:left
        rightMatrix:right resultMatrix:output];
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

struct RmsNormParamsHost {
    uint32_t rows, cols;
    float epsilon;
};

static void encodeRmsNorm(GraphState& state, const jint* cmd, int pos) {
    RmsNormParamsHost params{(uint32_t)positiveCount(cmd[pos + 6], "rms norm rows"),
                             (uint32_t)positiveCount(cmd[pos + 7], "rms norm columns"),
                             floatFromBits(cmd[pos + 8])};
    id<MTLBuffer> values = valueBuffer(state.ctx, &params, sizeof(params));
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->rmsNormPSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "rms norm") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "rms norm") offset:0 atIndex:1];
    [encoder setBuffer:requireBuffer(cmd[pos + 3], "rms norm") offset:0 atIndex:2];
    [encoder setBuffer:requireBuffer(cmd[pos + 4], "rms norm") offset:0 atIndex:3];
    [encoder setBuffer:requireBuffer(cmd[pos + 5], "rms norm") offset:0 atIndex:4];
    [encoder setBuffer:values offset:0 atIndex:5];
    NSUInteger threads = rowReductionWidth(state.ctx->rmsNormPSO);
    [encoder dispatchThreads:MTLSizeMake(threads, params.rows, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static void encodeRmsNormBackward(GraphState& state, const jint* cmd, int pos) {
    NSUInteger rows = positiveCount(cmd[pos + 6], "rms norm rows");
    uint32_t dims[] = {(uint32_t)rows, (uint32_t)positiveCount(cmd[pos + 7], "rms norm columns")};
    id<MTLBuffer> values = valueBuffer(state.ctx, dims, sizeof(dims));
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->rmsNormBackwardPSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "rms norm backward") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "rms norm backward") offset:0 atIndex:1];
    [encoder setBuffer:requireBuffer(cmd[pos + 3], "rms norm backward") offset:0 atIndex:2];
    [encoder setBuffer:requireBuffer(cmd[pos + 4], "rms norm backward") offset:0 atIndex:3];
    [encoder setBuffer:requireBuffer(cmd[pos + 5], "rms norm backward") offset:0 atIndex:4];
    [encoder setBuffer:values offset:0 atIndex:5];
    NSUInteger threads = rowReductionWidth(state.ctx->rmsNormBackwardPSO);
    [encoder dispatchThreads:MTLSizeMake(threads, rows, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static void encodeSwiGlu(GraphState& state, const jint* cmd, int pos) {
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->swiGluPSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "SwiGLU") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "SwiGLU") offset:0 atIndex:1];
    [encoder setBuffer:requireBuffer(cmd[pos + 3], "SwiGLU") offset:0 atIndex:2];
    dispatch1D(encoder, state.ctx->swiGluPSO,
               positiveCount(cmd[pos + 4], "SwiGLU element count"));
}

static void encodeSwiGluBackward(GraphState& state, const jint* cmd, int pos) {
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->swiGluBackwardPSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "SwiGLU backward") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "SwiGLU backward") offset:0 atIndex:1];
    [encoder setBuffer:requireBuffer(cmd[pos + 3], "SwiGLU backward") offset:0 atIndex:2];
    [encoder setBuffer:requireBuffer(cmd[pos + 4], "SwiGLU backward") offset:0 atIndex:3];
    [encoder setBuffer:requireBuffer(cmd[pos + 5], "SwiGLU backward") offset:0 atIndex:4];
    dispatch1D(encoder, state.ctx->swiGluBackwardPSO,
               positiveCount(cmd[pos + 6], "SwiGLU element count"));
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

static bool isFiveWidthOp(int op) {
    return isBinaryOp(op) || isScalarOp(op) || isReductionOp(op) ||
           op == OP_SWIGLU || op == OP_TRANSPOSE || op == OP_SOFTMAX_ROWS ||
           op == OP_SUM_ABS || op == OP_SUM_SQUARES || op == OP_SUM_SCALAR ||
           op == OP_CAUSAL_MASK;
}

static int commandWidth(int op) {
    if (isUnaryOp(op) || op == OP_SUM_SQUARES_SCALAR) return 4;
    if (isFiveWidthOp(op)) return 5;
    if (isBroadcastOp(op) || op == OP_CLAMP ||
        op == OP_SOFTMAX_BACKWARD || op == OP_CROSS_ENTROPY_LOSS ||
        op == OP_CROSS_ENTROPY_GRADIENT) return 6;
    if (op == OP_MATMUL || op == OP_SCATTER_ADD_ROWS ||
        op == OP_SCATTER_ADD_ROWS_ATOMIC || op == OP_LAYERNORM_BACKWARD ||
        op == OP_SPLIT_HEADS || op == OP_MERGE_HEADS || op == OP_GATHER_ROWS ||
        op == OP_CAUSAL_SOFTMAX || op == OP_CROSS_ENTROPY_FUSED ||
        op == OP_SWIGLU_BACKWARD) return 7;
    if (op == OP_ROTARY || op == OP_RMS_NORM) return 9;
    if (op == OP_RMS_NORM_BACKWARD) return 8;
    if (op == OP_BATCHED_MATMUL) return 11;
    if (op == OP_ADAMW_UPDATE) return 13;
    throw std::runtime_error("Unknown op code in graph: " + std::to_string(op));
}

static bool encodeBasicCommand(GraphState& state, const jint* cmd, int pos) {
    int op = cmd[pos];
    if (isBinaryOp(op)) encodeBinary(state, cmd, pos);
    else if (isUnaryOp(op)) encodeUnary(state, cmd, pos);
    else if (isScalarOp(op)) encodeScalar(state, cmd, pos);
    else if (isBroadcastOp(op)) encodeBroadcast(state, cmd, pos);
    else if (isReductionOp(op)) encodeReduction(state, cmd, pos);
    else if (op == OP_CLAMP) encodeClamp(state, cmd, pos);
    else if (op == OP_TRANSPOSE) encodeTranspose(state, cmd, pos);
    else if (op == OP_SPLIT_HEADS || op == OP_MERGE_HEADS) encodeHeadPermutation(state, cmd, pos);
    else if (op == OP_CAUSAL_MASK) encodeCausalMask(state, cmd, pos);
    else if (op == OP_CAUSAL_SOFTMAX) encodeCausalSoftmax(state, cmd, pos);
    else if (op == OP_ROTARY) encodeRotary(state, cmd, pos);
    else return false;
    return true;
}

static bool encodeTrainingCommand(GraphState& state, const jint* cmd, int pos) {
    int op = cmd[pos];
    if (op == OP_LAYERNORM_BACKWARD) encodeLayerNormBackward(state, cmd, pos);
    else if (op == OP_RMS_NORM) encodeRmsNorm(state, cmd, pos);
    else if (op == OP_RMS_NORM_BACKWARD) encodeRmsNormBackward(state, cmd, pos);
    else if (op == OP_SWIGLU) encodeSwiGlu(state, cmd, pos);
    else if (op == OP_SWIGLU_BACKWARD) encodeSwiGluBackward(state, cmd, pos);
    else if (op == OP_ADAMW_UPDATE) encodeAdamW(state, cmd, pos);
    else return false;
    return true;
}

static bool encodeSpecialCommand(GraphState& state, const jint* cmd, int pos) {
    int op = cmd[pos];
    if (op == OP_SCATTER_ADD_ROWS || op == OP_SCATTER_ADD_ROWS_ATOMIC) encodeScatter(state, cmd, pos);
    else if (op == OP_GATHER_ROWS) encodeGatherRows(state, cmd, pos);
    else if (op == OP_SUM_ABS || op == OP_SUM_SQUARES) encodeRowStatistic(state, cmd, pos);
    else if (op == OP_SUM_SQUARES_SCALAR) encodeSumSquaresScalar(state, cmd, pos);
    else if (op == OP_SUM_SCALAR) encodeScalarSum(state, cmd, pos);
    else if (op == OP_CROSS_ENTROPY_LOSS || op == OP_CROSS_ENTROPY_GRADIENT) encodeCrossEntropy(state, cmd, pos);
    else if (op == OP_CROSS_ENTROPY_FUSED) encodeFusedCrossEntropy(state, cmd, pos);
    else if (op == OP_MATMUL) encodeMatmul(state, cmd, pos);
    else if (op == OP_BATCHED_MATMUL) encodeBatchedMatmul(state, cmd, pos);
    else if (op == OP_SOFTMAX_ROWS) encodeSoftmax(state, cmd, pos);
    else if (op == OP_SOFTMAX_BACKWARD) encodeSoftmaxBackward(state, cmd, pos);
    else return encodeTrainingCommand(state, cmd, pos);
    return true;
}

static void encodeCommand(GraphState& state, const jint* cmd, int pos) {
    if (encodeBasicCommand(state, cmd, pos)) return;
    if (encodeSpecialCommand(state, cmd, pos)) return;
    throw std::runtime_error("Unknown op code in graph: " + std::to_string(cmd[pos]));
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
