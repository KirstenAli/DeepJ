#include <jni.h>
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

struct MetalContext {
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary>      library;

    id<MTLComputePipelineState> addPSO;
    id<MTLComputePipelineState> storageCopyPSO;
    id<MTLComputePipelineState> layoutTransferPSO;
    id<MTLComputePipelineState> layoutBinaryPSO;
    id<MTLComputePipelineState> layoutUnaryPSO;
    id<MTLComputePipelineState> layoutAssignPSO;
    id<MTLComputePipelineState> layoutReducePSO;
    id<MTLComputePipelineState> layoutSoftmaxPSO;
    id<MTLComputePipelineState> layoutMaximumBackwardPSO;
    id<MTLComputePipelineState> layoutMatmulPSO;
    id<MTLComputePipelineState> layoutIndexSelectPSO;
    id<MTLComputePipelineState> layoutIndexAddPSO;
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
    id<MTLComputePipelineState> accumulateSquaresPSO;
    id<MTLComputePipelineState> crossEntropyLossPSO;
    id<MTLComputePipelineState> crossEntropyGradPSO;
    id<MTLComputePipelineState> crossEntropyFusedPSO;
    id<MTLComputePipelineState> clampPSO;
    id<MTLComputePipelineState> powPSO;
    id<MTLComputePipelineState> scatterAddRowsPSO;
    id<MTLComputePipelineState> scatterLinksPSO;
    id<MTLComputePipelineState> orderedScatterPSO;
    id<MTLComputePipelineState> gatherRowsPSO;
    id<MTLComputePipelineState> headCacheWritePSO;
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
    id<MTLComputePipelineState> attentionPSO;
    id<MTLComputePipelineState> attentionDeltaPSO;
    id<MTLComputePipelineState> attentionQueryBackwardPSO;
    id<MTLComputePipelineState> attentionKeyValueBackwardPSO;
};
static NSString* tiledAttentionShaderSource = @R"(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct AttentionParams {
    uint length;
    uint groupSize;
    float scale;
};

struct AttentionInputs {
    device const float* query;
    device const float* key;
    device const float* value;
};

struct AttentionBackwardInputs {
    AttentionInputs values;
    device const float* gradient;
    device const float* normalization;
    device const float* delta;
};

struct AttentionTile {
    uint head;
    uint query;
    uint key;
};

// All matrix operands, accumulators and normalization statistics remain FP32.
template<uint Keys>
inline void attention_scores(device const float* query, device const float* key,
                              threadgroup float* scores, uint group) {
    for (uint column = 0; column < Keys; column += 8) {
        simdgroup_float8x8 result(0.0f);
        for (uint inner = 0; inner < 64; inner += 8) {
            simdgroup_float8x8 left, right;
            simdgroup_load(left, query + group * 8 * 64 + inner, 64);
            simdgroup_load(right, key + column * 64 + inner, 64, ulong2(0), true);
            simdgroup_multiply_accumulate(result, left, right, result);
        }
        simdgroup_store(result, scores + group * 8 * Keys + column, Keys);
    }
}

template<uint Keys>
inline void attention_values(threadgroup float* probabilities, device const float* values,
                              threadgroup float* output, uint group) {
    for (uint column = 0; column < 64; column += 8) {
        simdgroup_float8x8 result;
        simdgroup_load(result, output + group * 8 * 64 + column, 64);
        for (uint inner = 0; inner < Keys; inner += 8) {
            simdgroup_float8x8 left, right;
            simdgroup_load(left, probabilities + group * 8 * Keys + inner, Keys);
            simdgroup_load(right, values + inner * 64 + column, 64);
            simdgroup_multiply_accumulate(result, left, right, result);
        }
        simdgroup_store(result, output + group * 8 * 64 + column, 64);
    }
}

inline float attention_maximum(threadgroup float* scores, uint row, AttentionTile tile,
                                float scale, float maximum) {
    for (uint column = 0; column < 32; column++) {
        uint index = row * 32 + column;
        float value = tile.key + column <= tile.query + row ? scores[index] * scale : -INFINITY;
        scores[index] = value;
        maximum = max(maximum, value);
    }
    return maximum;
}

inline float attention_exponents(threadgroup float* scores, uint row, float maximum) {
    float sum = 0.0f;
    for (uint column = 0; column < 32; column++) {
        uint index = row * 32 + column;
        float value = exp(scores[index] - maximum);
        scores[index] = value;
        sum += value;
    }
    return sum;
}

inline void attention_statistics(threadgroup float* scores,
                                 threadgroup float* maxima, threadgroup float* sums,
                                 uint row, AttentionTile tile, float scale) {
    float maximum = attention_maximum(scores, row, tile, scale, maxima[row]);
    float correction = exp(maxima[row] - maximum);
    sums[row] = sums[row] * correction + attention_exponents(scores, row, maximum);
    maxima[row] = maximum;
}


inline void attention_online_statistics(threadgroup float* scores, threadgroup float* output,
                                         threadgroup float* maxima, threadgroup float* sums,
                                         uint row, AttentionTile tile, float scale) {
    float previous = maxima[row];
    attention_statistics(scores, maxima, sums, row, tile, scale);
    float correction = exp(previous - maxima[row]);
    for (uint column = 0; column < 64; column++) {
        output[row * 64 + column] *= correction;
    }
}

inline void attention_zero(threadgroup float* values, uint count, uint tid, uint threads) {
    for (uint index = tid; index < count; index += threads) {
        values[index] = 0.0f;
    }
}

inline void attention_initialize(threadgroup float* output, threadgroup float* maxima,
                                  threadgroup float* sums, uint tid) {
    attention_zero(output, 32 * 64, tid, 128);
    if (tid < 32) {
        maxima[tid] = -INFINITY;
        sums[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void attention_write(device float* output, device float* normalization,
                             threadgroup float* accumulated, threadgroup float* maxima,
                             threadgroup float* sums, AttentionTile tile, uint length, uint tid) {
    uint row = tile.head * length + tile.query;
    for (uint index = tid; index < 32 * 64; index += 128) {
        output[row * 64 + index] = accumulated[index] / sums[index / 64];
    }
    if (tid < 32) {
        normalization[(row + tid) * 2] = maxima[tid];
        normalization[(row + tid) * 2 + 1] = sums[tid];
    }
}


inline void attention_forward(AttentionInputs inputs, AttentionParams params, uint2 block,
                               uint tid, uint group, threadgroup float* scores,
                               threadgroup float* output, threadgroup float* maxima,
                               threadgroup float* sums) {
    uint start = block.x * 32, base = block.y * params.length * 64;
    uint keyBase = (block.y / params.groupSize) * params.length * 64;
    attention_initialize(output, maxima, sums, tid);
    for (uint key = 0; key <= start; key += 32) {
        attention_scores<32>(inputs.query + base + start * 64, inputs.key + keyBase + key * 64, scores, group);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 32) attention_online_statistics(scores, output, maxima, sums, tid,
                                                  AttentionTile{block.y, start, key}, params.scale);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        attention_values<32>(scores, inputs.value + keyBase + key * 64, output, group);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void kernel_tiled_attention(device const float* query [[buffer(0)]],
                                   device const float* key [[buffer(1)]],
                                   device const float* value [[buffer(2)]],
                                   device float* output [[buffer(3)]],
                                   constant AttentionParams& params [[buffer(4)]],
                                   device float* normalization [[buffer(5)]],
                                   uint2 block [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scores[32 * 32], accumulated[32 * 64], maxima[32], sums[32];
    attention_forward(AttentionInputs{query, key, value}, params, block, tid, group,
                       scores, accumulated, maxima, sums);
    attention_write(output, normalization, accumulated, maxima, sums,
                     AttentionTile{block.y, block.x * 32, 0}, params.length, tid);
}

kernel void kernel_attention_delta(device const float* output [[buffer(0)]],
                                   device const float* gradient [[buffer(1)]],
                                   device float* delta [[buffer(2)]],
                                   uint2 position [[thread_position_in_grid]]) {
    uint base = position.y * 64 + position.x;
    float value = output[base] * gradient[base] + output[base + 32] * gradient[base + 32];
    float sum = simd_sum(value);
    if (position.x == 0) delta[position.y] = sum;
}

inline void attention_derivatives(AttentionBackwardInputs inputs, AttentionParams params,
                                   AttentionTile tile, uint tid, threadgroup float* scores,
                                   threadgroup float* derivatives) {
    for (uint flat = tid; flat < 16 * 16; flat += 64) {
        uint row = flat / 16, column = flat % 16;
        uint index = tile.head * params.length + tile.query + row;
        float maximum = inputs.normalization[index * 2];
        float denominator = inputs.normalization[index * 2 + 1];
        float probability = tile.key + column <= tile.query + row
                ? exp(scores[flat] * params.scale - maximum) / denominator : 0.0f;
        scores[flat] = probability;
        // The first causal row has one choice, so its query/key derivative is exactly zero.
        derivatives[flat] = tile.query + row == 0 ? 0.0f
                : probability * (derivatives[flat] - inputs.delta[index]) * params.scale;
    }
}

inline void attention_backward_tile(AttentionBackwardInputs inputs, AttentionParams params,
                                     AttentionTile tile, uint tid, uint group,
                                     threadgroup float* scores, threadgroup float* derivatives) {
    uint query = (tile.head * params.length + tile.query) * 64;
    uint key = ((tile.head / params.groupSize) * params.length + tile.key) * 64;
    attention_scores<16>(inputs.values.query + query, inputs.values.key + key, scores, group);
    attention_scores<16>(inputs.gradient + query, inputs.values.value + key, derivatives, group);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    attention_derivatives(inputs, params, tile, tid, scores, derivatives);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void attention_transposed_values(threadgroup float* left, device const float* right,
                                         threadgroup float* output, uint group) {
    for (uint column = 0; column < 64; column += 8) {
        simdgroup_float8x8 result;
        simdgroup_load(result, output + group * 8 * 64 + column, 64);
        for (uint inner = 0; inner < 16; inner += 8) {
            simdgroup_float8x8 a, b;
            simdgroup_load(a, left + inner * 16 + group * 8, 16, ulong2(0), true);
            simdgroup_load(b, right + inner * 64 + column, 64);
            simdgroup_multiply_accumulate(result, a, b, result);
        }
        simdgroup_store(result, output + group * 8 * 64 + column, 64);
    }
}

inline void attention_query_gradient(AttentionBackwardInputs inputs, AttentionParams params,
                                      uint2 block, uint tid, uint group, threadgroup float* scores,
                                      threadgroup float* derivatives, threadgroup float* accumulated) {
    uint start = block.x * 16, base = (block.y / params.groupSize) * params.length * 64;
    attention_zero(accumulated, 16 * 64, tid, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint key = 0; key <= start; key += 16) {
        attention_backward_tile(inputs, params, AttentionTile{block.y, start, key}, tid, group,
                                 scores, derivatives);
        attention_values<16>(derivatives, inputs.values.key + base + key * 64, accumulated, group);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

inline void attention_write_gradient(device float* output, threadgroup float* values,
                                      uint2 block, uint length, uint tid) {
    uint base = (block.y * length + block.x * 16) * 64;
    for (uint index = tid; index < 16 * 64; index += 64) {
        output[base + index] = values[index];
    }
}

kernel void kernel_attention_query_backward(device const float* query [[buffer(0)]],
                                            device const float* key [[buffer(1)]],
                                            device const float* value [[buffer(2)]],
                                            device const float* gradient [[buffer(3)]],
                                            device const float* normalization [[buffer(4)]],
                                            device const float* delta [[buffer(5)]],
                                            device float* result [[buffer(6)]],
                                            constant AttentionParams& params [[buffer(7)]],
                                            uint2 block [[threadgroup_position_in_grid]],
                                            uint tid [[thread_index_in_threadgroup]],
                                            uint group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scores[16 * 16], derivatives[16 * 16], accumulated[16 * 64];
    AttentionBackwardInputs inputs = {{query, key, value}, gradient, normalization, delta};
    attention_query_gradient(inputs, params, block, tid, group, scores, derivatives, accumulated);
    attention_write_gradient(result, accumulated, block, params.length, tid);
}

inline void attention_accumulate_key_value(AttentionBackwardInputs inputs, AttentionParams params,
                                            AttentionTile tile, uint tid, uint group,
                                            threadgroup float* scores, threadgroup float* derivatives,
                                            threadgroup float* dk, threadgroup float* dv) {
    uint start = tile.key, base = tile.head * params.length * 64;
    for (uint query = start; query < params.length; query += 16) {
        attention_backward_tile(inputs, params, AttentionTile{tile.head, query, start}, tid, group,
                                 scores, derivatives);
        attention_transposed_values(derivatives, inputs.values.query + base + query * 64, dk, group);
        attention_transposed_values(scores, inputs.gradient + base + query * 64, dv, group);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

inline void attention_key_value_gradients(AttentionBackwardInputs inputs, AttentionParams params,
                                          uint2 block, uint tid, uint group,
                                          threadgroup float* scores, threadgroup float* derivatives,
                                          threadgroup float* dk, threadgroup float* dv) {
    attention_zero(dk, 16 * 64, tid, 64);
    attention_zero(dv, 16 * 64, tid, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint shared = 0; shared < params.groupSize; shared++) {
        AttentionTile tile = {block.y * params.groupSize + shared, 0, block.x * 16};
        attention_accumulate_key_value(inputs, params, tile, tid, group, scores, derivatives, dk, dv);
    }
}

kernel void kernel_attention_key_value_backward(device const float* query [[buffer(0)]],
                                                device const float* key [[buffer(1)]],
                                                device const float* value [[buffer(2)]],
                                                device const float* gradient [[buffer(3)]],
                                                device const float* normalization [[buffer(4)]],
                                                device const float* delta [[buffer(5)]],
                                                device float* dk [[buffer(6)]],
                                                device float* dv [[buffer(7)]],
                                                constant AttentionParams& params [[buffer(8)]],
                                                uint2 block [[threadgroup_position_in_grid]],
                                                uint tid [[thread_index_in_threadgroup]],
                                                uint group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scores[16 * 16], derivatives[16 * 16], keys[16 * 64], values[16 * 64];
    AttentionBackwardInputs inputs = {{query, key, value}, gradient, normalization, delta};
    attention_key_value_gradients(inputs, params, block, tid, group, scores, derivatives, keys, values);
    attention_write_gradient(dk, keys, block, params.length, tid);
    attention_write_gradient(dv, values, block, params.length, tid);
}
)";
static MetalContext* gCtx = nullptr;
static std::mutex gContextMutex;

static NSString* metalShaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

inline uint storage_round_shift(uint value, uint shift) {
    uint quotient = value >> shift;
    uint halfway = 1u << (shift - 1);
    uint remainder = value & ((1u << shift) - 1);
    return quotient + uint(remainder > halfway || (remainder == halfway && (quotient & 1u) != 0));
}

inline uint storage_decode_half(ushort bits) {
    uint sign = (uint(bits) & 0x8000u) << 16;
    uint exponent = (uint(bits) >> 10) & 31u;
    uint fraction = uint(bits) & 1023u;
    if (exponent == 31) return sign | 0x7f800000u | (fraction << 13);
    if (exponent != 0) return sign | ((exponent + 112u) << 23) | (fraction << 13);
    if (fraction == 0) return sign;
    uint shift = clz(fraction) - 21u;
    return sign | ((113u - shift) << 23) | (((fraction << shift) & 1023u) << 13);
}

inline ushort storage_encode_half(uint bits) {
    uint sign = (bits >> 16) & 0x8000u;
    uint fraction = bits & 0x007fffffu;
    int exponent = int((bits >> 23) & 255u) - 127;
    if (exponent == 128) return ushort(sign | 0x7c00u | (fraction == 0 ? 0u : 0x0200u | (fraction >> 13)));
    if (exponent > 15) return ushort(sign | 0x7c00u);
    if (exponent >= -14) {
        return ushort(sign | ((uint(exponent + 15) << 10) + storage_round_shift(fraction, 13)));
    }
    if (exponent < -25) return ushort(sign);
    return ushort(sign | storage_round_shift(fraction | 0x00800000u, uint(-exponent - 1)));
}

inline ushort storage_encode_bfloat(uint bits) {
    if ((bits & 0x7fffffffu) > 0x7f800000u) return ushort((bits >> 16) | 0x0040u);
    return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

inline uint storage_float_bits(device const unsigned char* input, uint index, uint dtype) {
    if (dtype == 1) return ((device const uint*)input)[index];
    ushort bits = ((device const ushort*)input)[index];
    return dtype == 2 ? storage_decode_half(bits) : uint(bits) << 16;
}

inline void storage_write_bits(device unsigned char* output, uint index, uint dtype, uint bits) {
    if (dtype == 1) ((device uint*)output)[index] = bits;
    else ((device ushort*)output)[index] = dtype == 2 ? storage_encode_half(bits) : storage_encode_bfloat(bits);
}

inline uint layout_address(device const ulong* words, uint start, uint logical) {
    uint rank = uint(words[start]);
    ulong address = words[start + 3];
    for (int axis = int(rank) - 1; axis >= 0; axis--) {
        uint dimension = uint(words[start + 4 + axis]);
        address += words[start + 4 + rank + axis] * (logical % dimension);
        logical /= dimension;
    }
    return uint(address);
}

inline void layout_raw_copy(device const unsigned char* input, device unsigned char* output,
                            uint source, uint target, uint dtype) {
    uint width = dtype == 1 ? 4u : 2u;
    for (uint byte = 0; byte < width; byte++) {
        output[ulong(target) * width + byte] = input[ulong(source) * width + byte];
    }
}

kernel void kernel_layout_transfer(device const unsigned char* input [[buffer(0)]],
                                   device unsigned char* output [[buffer(1)]],
                                   device const ulong* layouts [[buffer(2)]],
                                   constant uint4& params [[buffer(3)]],
                                   uint id [[thread_position_in_grid]]) {
    uint address = layout_address(layouts, params.x, id);
    if (params.y == params.z) layout_raw_copy(input, output, address, id, params.y);
    else storage_write_bits(output, id, params.z, storage_float_bits(input, address, params.y));
}

inline float layout_matmul_dot(device const unsigned char* a, device const unsigned char* b,
                               device const ulong* layouts, constant uint* params, uint index) {
    uint m = params[5], n = params[6], k = params[7], batch = index / (m * n);
    uint first = layout_address(layouts, params[0], (batch * m + index / n % m) * k);
    uint second = layout_address(layouts, params[1], batch * k * n + index % n);
    uint aRank = uint(layouts[params[0]]), bRank = uint(layouts[params[1]]);
    ulong aStep = layouts[params[0] + 3 + 2 * aRank];
    ulong bStep = layouts[params[1] + 2 + 2 * bRank];
    float sum = 0.0f;
    for (uint inner = 0; inner < k; inner++) {
        float x = as_type<float>(storage_float_bits(a, uint(first + inner * aStep), params[2]));
        float y = as_type<float>(storage_float_bits(b, uint(second + inner * bStep), params[3]));
        sum += x * y;
    }
    return sum;
}

kernel void kernel_layout_matmul(device const unsigned char* a [[buffer(0)]],
                                 device const unsigned char* b [[buffer(1)]],
                                 device unsigned char* output [[buffer(2)]],
                                 device const ulong* layouts [[buffer(3)]],
                                 constant uint* params [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    float sum = layout_matmul_dot(a, b, layouts, params, index);
    storage_write_bits(output, index, params[4], as_type<uint>(sum));
}

kernel void kernel_layout_index_select(device const unsigned char* input [[buffer(0)]],
                                       device unsigned char* output [[buffer(1)]],
                                       device const ulong* layouts [[buffer(2)]],
                                       constant uint* params [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    uint row = index / params[3], selected = uint(layouts[params[1] + 2 + row % params[4]]);
    uint logical = (row / params[4] * params[2] + selected) * params[3] + index % params[3];
    layout_raw_copy(input, output, layout_address(layouts, params[0], logical), index, params[5]);
}

inline float layout_index_sum(device const unsigned char* updates, device const ulong* layouts,
                               constant uint* params, uint other, uint group, float sum) {
    uint start = params[2], count = uint(layouts[start]), groups = params[7];
    uint offsets = start + 2 + count + groups, positions = offsets + groups + 1;
    uint first = uint(layouts[offsets + group]), end = uint(layouts[offsets + group + 1]);
    uint base = other / params[4] * count * params[4] + other % params[4];
    for (uint position = first; position < end; position++) {
        uint row = uint(layouts[positions + position]);
        uint address = layout_address(layouts, params[1], base + row * params[4]);
        sum += as_type<float>(storage_float_bits(updates, address, params[6]));
    }
    return sum;
}

kernel void kernel_layout_index_add(device unsigned char* target [[buffer(0)]],
                                    device const unsigned char* updates [[buffer(1)]],
                                    device const ulong* layouts [[buffer(2)]],
                                    constant uint* params [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    uint other = index / params[7], group = index % params[7], count = uint(layouts[params[2]]);
    uint selected = uint(layouts[params[2] + 2 + count + group]);
    uint logical = (other / params[4] * params[3] + selected) * params[4] + other % params[4];
    uint address = layout_address(layouts, params[0], logical);
    float initial = as_type<float>(storage_float_bits(target, address, params[5]));
    float sum = layout_index_sum(updates, layouts, params, other, group, initial);
    storage_write_bits(target, address, params[5], as_type<uint>(sum));
}

inline float layout_gelu(float x) {
    float t = 0.7978845608f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanh(t));
}

inline float layout_gelu_gradient(float x, float gradient) {
    float x2 = x * x, t = 0.7978845608f * (x + 0.044715f * x2 * x);
    float tanhT = tanh(t), sech2 = 1.0f - tanhT * tanhT;
    float derivative = 0.7978845608f * (1.0f + 3.0f * 0.044715f * x2);
    return gradient * (0.5f * (1.0f + tanhT) + 0.5f * x * sech2 * derivative);
}

inline float layout_binary_value(float a, float b, uint operation) {
    switch (operation) {
        case 1: return a + b;
        case 2: return a - b;
        case 3: return a * b;
        case 4: return a / b;
        case 14: return a > 0.0f ? b : 0.0f;
        default: return layout_gelu_gradient(a, b);
    }
}

inline float layout_activation_value(float x, uint operation) {
    switch (operation) {
        case 9: return exp(x);
        case 10: return log(x);
        case 11: return tanh(x);
        case 12: return 1.0f / (1.0f + exp(-x));
        case 13: return max(0.0f, x);
        default: return layout_gelu(x);
    }
}

inline float layout_unary_value(float x, uint operation, float first, float second) {
    switch (operation) {
        case 6: return x * first;
        case 7: return sqrt(x);
        case 8: return -x;
        case 21: return x + first;
        case 22: return x / first;
        case 35: return min(second, max(first, x));
        case 36: return pow(x, first);
        default: return layout_activation_value(x, operation);
    }
}

kernel void kernel_layout_binary(device const unsigned char* a [[buffer(0)]],
                                 device const unsigned char* b [[buffer(1)]],
                                 device unsigned char* output [[buffer(2)]],
                                 device const ulong* layouts [[buffer(3)]],
                                 constant uint* params [[buffer(4)]],
                                 uint index [[thread_position_in_grid]]) {
    uint first = layout_address(layouts, params[0], index), second = layout_address(layouts, params[1], index);
    float left = as_type<float>(storage_float_bits(a, first, params[2]));
    float right = as_type<float>(storage_float_bits(b, second, params[3]));
    float result = layout_binary_value(left, right, params[5]);
    storage_write_bits(output, index, params[4], as_type<uint>(result));
}

kernel void kernel_layout_unary(device const unsigned char* input [[buffer(0)]],
                                device unsigned char* output [[buffer(1)]],
                                device const ulong* layouts [[buffer(2)]],
                                constant uint* params [[buffer(3)]],
                                uint index [[thread_position_in_grid]]) {
    uint address = layout_address(layouts, params[0], index);
    float value = as_type<float>(storage_float_bits(input, address, params[1]));
    float result = layout_unary_value(value, params[2], as_type<float>(params[3]), as_type<float>(params[4]));
    storage_write_bits(output, index, params[1], as_type<uint>(result));
}

kernel void kernel_layout_assign(device const unsigned char* input [[buffer(0)]],
                                 device unsigned char* output [[buffer(1)]],
                                 device const ulong* layouts [[buffer(2)]],
                                 constant uint4& params [[buffer(3)]],
                                 uint index [[thread_position_in_grid]]) {
    uint source = layout_address(layouts, params.x, index), target = layout_address(layouts, params.y, index);
    if (params.z == params.w) layout_raw_copy(input, output, source, target, params.z);
    else storage_write_bits(output, target, params.w, storage_float_bits(input, source, params.z));
}

inline float layout_value(device const unsigned char* input, device const ulong* layouts,
                          uint start, uint dtype, uint logical) {
    return as_type<float>(storage_float_bits(input, layout_address(layouts, start, logical), dtype));
}

inline bool layout_nan(float value) {
    return (as_type<uint>(value) & 0x7fffffffu) > 0x7f800000u;
}

inline float layout_maximum(float first, float second) {
    if (layout_nan(first) || layout_nan(second)) return as_type<float>(0x7fc00000u);
    if (first == 0.0f && second == 0.0f) return as_type<float>(as_type<uint>(first) & as_type<uint>(second));
    return first > second ? first : second;
}

inline float layout_group_sum(device const unsigned char* input, device const ulong* layouts,
                              uint start, uint dtype, uint base, uint width, uint operation) {
    float sum = 0.0f;
    for (uint index = 0; index < width; index++) {
        float value = layout_value(input, layouts, start, dtype, base + index);
        if (operation == 4) value = abs(value);
        if (operation == 5) value *= value;
        sum += value;
    }
    return sum;
}

inline float layout_group_maximum(device const unsigned char* input, device const ulong* layouts,
                                  uint start, uint dtype, uint base, uint width) {
    float maximum = -INFINITY;
    for (uint index = 0; index < width; index++) {
        maximum = layout_maximum(maximum, layout_value(input, layouts, start, dtype, base + index));
    }
    return maximum;
}

inline float layout_group_variance(device const unsigned char* input, device const ulong* layouts,
                                   constant uint* params, uint base, float mean) {
    if (params[4] <= params[6]) return as_type<float>(0x7fc00000u);
    float sum = 0.0f;
    for (uint index = 0; index < params[4]; index++) {
        float centered = layout_value(input, layouts, params[0], params[1], base + index) - mean;
        sum += centered * centered;
    }
    return sum / float(params[4] - params[6]);
}

inline float layout_reduced_value(device const unsigned char* input, device const ulong* layouts,
                                  constant uint* params, uint group) {
    uint base = group * params[4];
    if (params[5] == 3) return layout_group_maximum(input, layouts, params[0], params[1], base, params[4]);
    float sum = layout_group_sum(input, layouts, params[0], params[1], base, params[4], params[5]);
    if (params[5] == 1) return sum / float(params[4]);
    if (params[5] == 2) return layout_group_variance(input, layouts, params, base, sum / float(params[4]));
    return sum;
}

kernel void kernel_layout_reduce(device const unsigned char* input [[buffer(0)]],
                                 device unsigned char* output [[buffer(1)]],
                                 device const ulong* layouts [[buffer(2)]],
                                 constant uint* params [[buffer(3)]],
                                 uint group [[thread_position_in_grid]]) {
    float value = layout_reduced_value(input, layouts, params, group);
    storage_write_bits(output, group, params[2], as_type<uint>(value));
}

inline float layout_softmax_sum(device const unsigned char* input, device const ulong* layouts,
                                constant uint* params, uint base, float maximum) {
    float sum = 0.0f;
    for (uint index = 0; index < params[3]; index++) {
        sum += exp(layout_value(input, layouts, params[0], params[2], base + index) - maximum);
    }
    return sum;
}

inline void layout_softmax_write(device const unsigned char* input, device unsigned char* output,
                                 device const ulong* layouts, constant uint* params, uint base,
                                 float maximum, float sum) {
    bool invalid = (as_type<uint>(maximum) & 0x7fffffffu) >= 0x7f800000u;
    for (uint index = 0; index < params[3]; index++) {
        float value = invalid ? as_type<float>(0x7fc00000u)
                : exp(layout_value(input, layouts, params[0], params[2], base + index) - maximum) / sum;
        uint address = layout_address(layouts, params[1], base + index);
        storage_write_bits(output, address, params[2], as_type<uint>(value));
    }
}

kernel void kernel_layout_softmax(device const unsigned char* input [[buffer(0)]],
                                  device unsigned char* output [[buffer(1)]],
                                  device const ulong* layouts [[buffer(2)]],
                                  constant uint* params [[buffer(3)]],
                                  uint group [[thread_position_in_grid]]) {
    uint base = group * params[3];
    float maximum = layout_group_maximum(input, layouts, params[0], params[2], base, params[3]);
    float sum = layout_softmax_sum(input, layouts, params, base, maximum);
    layout_softmax_write(input, output, layouts, params, base, maximum, sum);
}

inline uint layout_maximum_ties(device const unsigned char* input, device const ulong* layouts,
                                constant uint* params, uint base, float maximum) {
    uint count = 0;
    for (uint index = 0; index < params[6]; index++) {
        if (layout_value(input, layouts, params[0], params[4], base + index) == maximum) count++;
    }
    return count;
}

inline float layout_maximum_derivative(float value, float maximum, float gradient, uint ties) {
    if (layout_nan(maximum)) return as_type<float>(0x7fc00000u);
    return value == maximum ? gradient / float(ties) : 0.0f;
}

inline void layout_maximum_write(device const unsigned char* input, device unsigned char* output,
                                 device const ulong* layouts, constant uint* params, uint base,
                                 float maximum, float gradient, uint ties) {
    for (uint index = 0; index < params[6]; index++) {
        float value = layout_value(input, layouts, params[0], params[4], base + index);
        float derivative = layout_maximum_derivative(value, maximum, gradient, ties);
        uint address = layout_address(layouts, params[3], base + index);
        storage_write_bits(output, address, 1, as_type<uint>(derivative));
    }
}

kernel void kernel_layout_maximum_backward(device const unsigned char* input [[buffer(0)]],
                                           device const unsigned char* maximum [[buffer(1)]],
                                           device const unsigned char* gradient [[buffer(2)]],
                                           device unsigned char* output [[buffer(3)]],
                                           device const ulong* layouts [[buffer(4)]],
                                           constant uint* params [[buffer(5)]],
                                           uint group [[thread_position_in_grid]]) {
    uint base = group * params[6];
    float winner = layout_value(maximum, layouts, params[1], params[5], group);
    float upstream = layout_value(gradient, layouts, params[2], 1, group);
    uint ties = layout_maximum_ties(input, layouts, params, base, winner);
    layout_maximum_write(input, output, layouts, params, base, winner, upstream, ties);
}

kernel void kernel_storage_copy(device const unsigned char* input [[buffer(0)]],
                                device unsigned char* output [[buffer(1)]],
                                constant uint4& range [[buffer(2)]],
                                uint id [[thread_position_in_grid]]) {
    ulong source = (ulong(range.y) + id) * range.w;
    ulong target = (ulong(range.z) + id) * range.w;
    for (uint byte = 0; byte < range.w; byte++) {
        output[target + byte] = input[source + byte];
    }
}

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
    for (uint c = tid; c < cols; c += width) {
        result += values[base + c];
    }
    return result;
}

inline float row_abs_sum(device const float* values, uint base, uint cols,
                         uint tid, uint width) {
    float result = 0.0f;
    for (uint c = tid; c < cols; c += width) {
        result += fabs(values[base + c]);
    }
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
    for (uint c = tid; c < cols; c += width) {
        result = max(result, values[base + c]);
    }
    return result;
}

inline float row_exp_sum(device const float* values, uint base, uint cols,
                         uint tid, uint width, float maximum) {
    float result = 0.0f;
    for (uint c = tid; c < cols; c += width) {
        result += exp(values[base + c] - maximum);
    }
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
    for (uint row = tid; row < rows; row += width) {
        result += values[row * cols + column];
    }
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
    // Every lane must read the result before a following reduction reuses scratch.
    float result = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

inline float reduce_max(threadgroup float* scratch, float value,
                        uint tid, uint width) {
    scratch[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = width >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float result = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

inline void write_row_value(device float* output, uint base, uint cols,
                            uint tid, uint width, float value) {
    for (uint c = tid; c < cols; c += width) {
        output[base + c] = value;
    }
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

struct HeadCacheWriteParams {
    uint heads, length, capacity, width, offset;
};

kernel void kernel_head_cache_write(device const float* update [[buffer(0)]],
                                    device float* cache [[buffer(1)]],
                                    constant HeadCacheWriteParams& params [[buffer(2)]],
                                    uint index [[thread_position_in_grid]]) {
    uint perHead = params.length * params.width;
    if (index >= params.heads * perHead) return;
    uint head = index / perHead, within = index % perHead;
    uint destination = (head * params.capacity + params.offset) * params.width + within;
    cache[destination] = update[index];
}

kernel void kernel_sum_squares_scalar(device const float* input [[buffer(0)]],
                                      device float* partials [[buffer(1)]],
                                      device const uint2* dims [[buffer(2)]],
                                      uint id [[thread_position_in_grid]],
                                      uint tid [[thread_position_in_threadgroup]],
                                      uint group [[threadgroup_position_in_grid]],
                                      uint width [[threads_per_threadgroup]]) {
    uint count = dims[0].x, gridWidth = dims[0].y * width;
    threadgroup float scratch[1024];
    float value = 0.0f;
    for (uint index = id; index < count; index += gridWidth) {
        value += input[index] * input[index];
    }
    float sum = reduce_sum(scratch, value, tid, width);
    if (tid == 0) partials[group] = sum;
}

kernel void kernel_accumulate_squares(device const float* partials [[buffer(0)]],
                                      device float* total [[buffer(1)]],
                                      device const uint* count [[buffer(2)]],
                                      uint tid [[thread_position_in_threadgroup]],
                                      uint width [[threads_per_threadgroup]]) {
    threadgroup float scratch[1024];
    float value = 0.0f;
    for (uint index = tid; index < count[0]; index += width) {
        value += partials[index];
    }
    float sum = reduce_sum(scratch, value, tid, width);
    if (tid == 0) total[0] += sum;
}

inline bool first_row_occurrence(device const float* indices, uint row, uint index) {
    for (uint previous = 0; previous < index; previous++) {
        if ((uint)indices[previous] == row) return false;
    }
    return true;
}

inline uint next_row_occurrence(device const float* indices, uint row,
                                 uint index, uint count) {
    for (uint next = index + 1; next < count; next++) {
        if ((uint)indices[next] == row) return next;
    }
    return count;
}

kernel void kernel_scatter_row_links(device const float* indices [[buffer(0)]],
                                     device uint2* links [[buffer(1)]],
                                     device const uint3* dims [[buffer(2)]],
                                     uint id [[thread_position_in_grid]]) {
    uint count = dims[0].z;
    if (id >= count) return;
    uint row = (uint)indices[id];
    links[id].x = first_row_occurrence(indices, row, id);
    links[id].y = next_row_occurrence(indices, row, id, count);
}

kernel void kernel_scatter_add_rows_ordered(device float* target        [[buffer(0)]],
                                           device const float* indices [[buffer(1)]],
                                           device const float* grad    [[buffer(2)]],
                                           device const uint3* dims    [[buffer(3)]],
                                           device const uint2* links   [[buffer(4)]],
                                           uint id [[thread_position_in_grid]]) {
    uint targetRows = dims[0].x, targetCols = dims[0].y, nIdx = dims[0].z;
    if (id >= nIdx * targetCols) return;
    uint i = id / targetCols, c = id % targetCols, row = (uint)indices[i];
    if (row >= targetRows || links[i].x == 0) return;
    uint flat = row * targetCols + c;
    float value = target[flat];
    for (uint index = i; index < nIdx; index = links[index].y) {
        value += grad[index * targetCols + c];
    }
    target[flat] = value;
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
    for (uint c = tid; c < cols; c += width) {
        result += left[base + c] * right[base + c];
    }
    return result;
}

inline void write_softmax_backward(device const float* gradient, device const float* softmax,
                                   device float* output, uint base, uint cols,
                                   uint tid, uint width, float dot, float scale) {
#pragma clang fp reassociate(off)
    // Keep the derivative's FP32 rounding before the formerly separate scale kernel.
    for (uint c = tid; c < cols; c += width) {
        float value = softmax[base + c];
        float derivative = value * (gradient[base + c] - dot);
        output[base + c] = derivative * scale;
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
    for (uint col = tid; col < cols; col += width) {
        output[base + col] /= sum;
    }
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
                                    constant float& scale          [[buffer(4)]],
                                    uint3 gid [[thread_position_in_grid]],
                                    uint3 tid3 [[thread_position_in_threadgroup]],
                                    uint3 tptg [[threads_per_threadgroup]]) {
    uint cols = dims[0], row = gid.y, tid = tid3.x;
    threadgroup float scratch[1024];
    uint base = row * cols;
    float local = row_dot(gradOutput, softmaxOut, base, cols, tid, tptg.x);
    float dot = reduce_sum(scratch, local, tid, tptg.x);
    write_softmax_backward(gradOutput, softmaxOut, out, base, cols, tid, tptg.x, dot, scale);
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
        throw std::runtime_error(std::string("Failed to create PSO for ")
                + [name UTF8String] + ": " + [desc UTF8String]);
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

static id<MTLLibrary> compileLibrary(id<MTLDevice> device, NSString* source = metalShaderSource) {
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source
                                                  options:makeCompileOptions()
                                                    error:&error];
    if (library != nil) return library;
    NSString* desc = error.localizedDescription ?: @"Unknown error";
    throw std::runtime_error(std::string("Failed to compile Metal shaders: ") + [desc UTF8String]);
}

static id<MTLLibrary> compileAttentionLibrary(id<MTLDevice> device) {
    if ([device supportsFamily:MTLGPUFamilyApple7]) return compileLibrary(device, tiledAttentionShaderSource);
    // Virtual devices may expose the instructions without advertising an Apple GPU family.
    return [device newLibraryWithSource:tiledAttentionShaderSource options:makeCompileOptions() error:nil];
}

static void initLayoutPipelines(MetalContext* ctx, id<MTLLibrary> library) {
    ctx->storageCopyPSO = makePSO(library, @"kernel_storage_copy");
    ctx->layoutTransferPSO = makePSO(library, @"kernel_layout_transfer");
    ctx->layoutBinaryPSO = makePSO(library, @"kernel_layout_binary");
    ctx->layoutUnaryPSO = makePSO(library, @"kernel_layout_unary");
    ctx->layoutAssignPSO = makePSO(library, @"kernel_layout_assign");
    ctx->layoutReducePSO = makePSO(library, @"kernel_layout_reduce");
    ctx->layoutSoftmaxPSO = makePSO(library, @"kernel_layout_softmax");
    ctx->layoutMaximumBackwardPSO = makePSO(library, @"kernel_layout_maximum_backward");
    ctx->layoutMatmulPSO = makePSO(library, @"kernel_layout_matmul");
    ctx->layoutIndexSelectPSO = makePSO(library, @"kernel_layout_index_select");
    ctx->layoutIndexAddPSO = makePSO(library, @"kernel_layout_index_add");
}

static void initBasicPipelines(MetalContext* ctx, id<MTLLibrary> library) {
    initLayoutPipelines(ctx, library);
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
    ctx->headCacheWritePSO = makePSO(library, @"kernel_head_cache_write");
}

static void initTiledAttentionPipelines(MetalContext* ctx) {
    id<MTLLibrary> library = compileAttentionLibrary(ctx->device);
    if (library == nil) return;
    ctx->attentionPSO = makePSO(library, @"kernel_tiled_attention");
    ctx->attentionDeltaPSO = makePSO(library, @"kernel_attention_delta");
    ctx->attentionQueryBackwardPSO = makePSO(library, @"kernel_attention_query_backward");
    ctx->attentionKeyValueBackwardPSO = makePSO(library, @"kernel_attention_key_value_backward");
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
    ctx->accumulateSquaresPSO = makePSO(library, @"kernel_accumulate_squares");
}

static void initLossPipelines(MetalContext* ctx, id<MTLLibrary> library) {
    ctx->crossEntropyLossPSO = makePSO(library, @"kernel_cross_entropy_loss");
    ctx->crossEntropyGradPSO = makePSO(library, @"kernel_cross_entropy_gradient");
    ctx->crossEntropyFusedPSO = makePSO(library, @"kernel_cross_entropy_fused");
    ctx->clampPSO = makePSO(library, @"kernel_clamp");
    ctx->powPSO = makePSO(library, @"kernel_pow");
    ctx->scatterAddRowsPSO = makePSO(library, @"kernel_scatter_add_rows");
    ctx->scatterLinksPSO = makePSO(library, @"kernel_scatter_row_links");
    ctx->orderedScatterPSO = makePSO(library, @"kernel_scatter_add_rows_ordered");
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
    initTiledAttentionPipelines(ctx);
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

extern "C" JNIEXPORT jboolean JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_nativeSupportsTiledAttention(
        JNIEnv* env, jclass) {
    try {
        return getContext()->attentionPSO != nil ? JNI_TRUE : JNI_FALSE;
    } catch (const std::exception& error) {
        throwJavaRuntimeException(env, error.what());
        return JNI_FALSE;
    }
}

#include <unordered_map>

using BufferPool = std::unordered_map<int, id<MTLBuffer>>;
using BufferEntry = BufferPool::iterator;

static BufferPool gBufferPool;
static std::unordered_map<int, int> gBufferDTypes;
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
static constexpr int OP_SCALED_SOFTMAX_BACKWARD = 57;
static constexpr int OP_TILED_CAUSAL_ATTENTION = 58;
static constexpr int OP_TILED_CAUSAL_ATTENTION_BACKWARD = 59;
static constexpr int OP_HEAD_CACHE_WRITE = 60;
static constexpr int OP_HEAD_PREFIX_MATMUL = 61;
static constexpr int OP_STORAGE_COPY = 62;
static constexpr int OP_LAYOUT_TRANSFER = 63;
static constexpr int OP_LAYOUT_BINARY = 64;
static constexpr int OP_LAYOUT_UNARY = 65;
static constexpr int OP_LAYOUT_ASSIGN = 66;
static constexpr int OP_LAYOUT_REDUCE = 67;
static constexpr int OP_LAYOUT_SOFTMAX = 68;
static constexpr int OP_LAYOUT_MAXIMUM_BACKWARD = 69;
static constexpr int OP_LAYOUT_MATMUL = 70;
static constexpr int OP_LAYOUT_INDEX_SELECT = 71;
static constexpr int OP_LAYOUT_INDEX_ADD = 72;

static id<MTLBuffer> requireStorageBuffer(int id, const char* opName) {
    auto it = gBufferPool.find(id);
    if (it == gBufferPool.end() || it->second == nil) {
        throw std::runtime_error(std::string(opName) + ": missing GPU buffer id=" + std::to_string(id));
    }
    return it->second;
}

static id<MTLBuffer> requireBuffer(int bufferId, const char* opName) {
    id<MTLBuffer> buffer = requireStorageBuffer(bufferId, opName);
    if (gBufferDTypes.at(bufferId) != 1) {
        throw std::runtime_error(std::string(opName) + ": matrix opcode requires FLOAT32 storage");
    }
    return buffer;
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

// Metal copies these transient arguments immediately; no caller-owned buffer is needed.
template<class T>
static void bindValue(id<MTLComputeCommandEncoder> encoder, const T& value, NSUInteger index) {
    static_assert(sizeof(T) < 4096, "Inline Metal arguments must be smaller than 4 KiB");
    static_assert(std::is_trivially_copyable_v<T> && !std::is_pointer_v<T>, "Pass argument values, not pointers");
    [encoder setBytes:&value length:sizeof(T) atIndex:index];
}

struct SoftmaxBuffers {
    id<MTLBuffer> maximum;
    id<MTLBuffer> sum;
    uint32_t columns;
};

static SoftmaxBuffers makeSoftmaxBuffers(MetalContext* ctx, int rows, int cols) {
    NSUInteger bytes = (NSUInteger)rows * sizeof(float);
    id<MTLBuffer> maximum = [ctx->device newBufferWithLength:bytes
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> sum = [ctx->device newBufferWithLength:bytes
                                                 options:MTLResourceStorageModeShared];
    if (maximum == nil || sum == nil) {
        throw std::runtime_error("Failed to allocate softmax buffers");
    }
    return SoftmaxBuffers{maximum, sum, (uint32_t)cols};
}

static void encodeSoftmaxMax(id<MTLComputeCommandEncoder> encoder, MetalContext* ctx,
                             id<MTLBuffer> input, const SoftmaxBuffers& buffers, int rows) {
    [encoder setComputePipelineState:ctx->softmaxMaxPSO];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:buffers.maximum offset:0 atIndex:1];
    bindValue(encoder, buffers.columns, 2);
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
    bindValue(encoder, buffers.columns, 4);
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
    bindValue(encoder, buffers.columns, 2);
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

static NSUInteger storageBytes(int elements, int dtype) {
    if (elements <= 0 || dtype < 1 || dtype > 3) throw std::runtime_error("Invalid typed GPU allocation");
    return (NSUInteger)elements * (dtype == 1 ? sizeof(float) : sizeof(jshort));
}

static void allocateTypedBuffer(MetalContext* ctx, int bufferId, int elements, int dtype) {
    if (bufferId < 0) throw std::runtime_error("Invalid GPU buffer id");
    NSUInteger bytes = storageBytes(elements, dtype);
    auto existing = gBufferPool.find(bufferId);
    if (existing != gBufferPool.end()) {
        if ([existing->second length] != bytes || gBufferDTypes.at(bufferId) != dtype) {
            throw std::runtime_error("GPU buffer id already has a different size or dtype");
        }
        return;
    }
    id<MTLBuffer> buffer = [ctx->device newBufferWithLength:bytes
                                                    options:MTLResourceStorageModeShared];
    if (buffer == nil) throw std::runtime_error("Failed to allocate Metal buffer");
    gBufferPool[bufferId] = buffer;
    gBufferDTypes[bufferId] = dtype;
}

static void allocateBuffers(const jint* ids, const jint* sizes, int count) {
    MetalContext* ctx = getContext();
    for (int i = 0; i < count; i++) {
        allocateTypedBuffer(ctx, ids[i], sizes[i], 1);
    }
}

static BufferEntry transferEntry(JNIEnv* env, int bufferId, jarray values,
                                 NSUInteger valueBytes, const char* operation) {
    if (values == nullptr) {
        throwJavaRuntimeException(env, (std::string(operation) + " data cannot be null").c_str());
        return gBufferPool.end();
    }
    auto entry = gBufferPool.find(bufferId);
    if (entry == gBufferPool.end()) {
        throwJavaRuntimeException(env, (std::string(operation) + " buffer was not found").c_str());
        return entry;
    }
    int dtype = gBufferDTypes.at(bufferId);
    NSUInteger bytes = (NSUInteger)env->GetArrayLength(values) * valueBytes;
    if (bytes == [entry->second length] && (dtype == 1) == (valueBytes == sizeof(float))) return entry;
    throwJavaRuntimeException(env, (std::string(operation) + " length or dtype does not match buffer").c_str());
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

struct IntArrayView {
    JNIEnv* env;
    jintArray source;
    jint* values;

    IntArrayView(JNIEnv* environment, jintArray array) : env(environment), source(array),
            values(env->GetIntArrayElements(source, nullptr)) {}
    ~IntArrayView() {
        if (values != nullptr) env->ReleaseIntArrayElements(source, values, JNI_ABORT);
    }
};

static void allocateRequestedTypedBuffers(JNIEnv* env, jintArray idsArr, jintArray sizesArr,
                                          jintArray typesArr, jint count) {
    if (!validateAllocationArrays(env, idsArr, sizesArr, count) ||
        !validateAllocationArrays(env, idsArr, typesArr, count)) return;
    IntArrayView ids(env, idsArr), sizes(env, sizesArr), types(env, typesArr);
    if (ids.values == nullptr || sizes.values == nullptr || types.values == nullptr) return;
    for (int index = 0; index < count; index++) {
        if (ids.values[index] < 0) throw std::runtime_error("Invalid GPU buffer id");
        storageBytes(sizes.values[index], types.values[index]);
    }
    std::lock_guard<std::mutex> lock(gBufferMutex);
    MetalContext* ctx = getContext();
    for (int index = 0; index < count; index++) {
        allocateTypedBuffer(ctx, ids.values[index], sizes.values[index], types.values[index]);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_nativeAllocTypedBuffers(
        JNIEnv* env, jclass, jintArray ids, jintArray elements, jintArray dtypes, jint count) {
    @autoreleasepool {
        try { allocateRequestedTypedBuffers(env, ids, elements, dtypes, count); }
        catch (const std::exception& error) { throwJavaRuntimeException(env, error.what()); }
    }
}

static void transferBuffer16(JNIEnv* env, jint bufferId, jshortArray values, bool upload) {
    std::lock_guard<std::mutex> lock(gBufferMutex);
    BufferEntry entry = transferEntry(env, bufferId, values, sizeof(jshort), "16-bit transfer");
    if (entry == gBufferPool.end()) return;
    jint length = env->GetArrayLength(values);
    jshort* memory = (jshort*)[entry->second contents];
    if (upload) env->GetShortArrayRegion(values, 0, length, memory);
    else env->SetShortArrayRegion(values, 0, length, memory);
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_nativeUploadBuffer16(
        JNIEnv* env, jclass, jint bufferId, jshortArray values) {
    @autoreleasepool { transferBuffer16(env, bufferId, values, true); }
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_nativeDownloadBuffer16(
        JNIEnv* env, jclass, jint bufferId, jshortArray values) {
    @autoreleasepool { transferBuffer16(env, bufferId, values, false); }
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_nativeBufferByteLength(
        JNIEnv* env, jclass, jint bufferId) {
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(gBufferMutex);
        auto entry = gBufferPool.find(bufferId);
        if (entry != gBufferPool.end()) return (jlong)[entry->second length];
        throwJavaRuntimeException(env, "GPU buffer was not found");
        return 0;
    }
}

static void uploadBuffer(JNIEnv* env, jint bufId, jfloatArray dataArr) {
    std::lock_guard<std::mutex> lock(gBufferMutex);
    BufferEntry entry = transferEntry(env, bufId, dataArr, sizeof(float), "Upload");
    if (entry == gBufferPool.end()) return;
    jint len = env->GetArrayLength(dataArr);
    env->GetFloatArrayRegion(dataArr, 0, len, (jfloat*)[entry->second contents]);
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_nativeUploadBuffer(
        JNIEnv* env, jclass, jint bufId, jfloatArray dataArr) {
    @autoreleasepool { uploadBuffer(env, bufId, dataArr); }
}

static void zeroTypedBuffer(JNIEnv* env, jint bufferId, jint elements, jint dtype) {
    std::lock_guard<std::mutex> lock(gBufferMutex);
    auto entry = gBufferPool.find(bufferId);
    if (entry == gBufferPool.end() || elements <= 0 || dtype < 1 || dtype > 3 ||
        gBufferDTypes.at(bufferId) != dtype || storageBytes(elements, dtype) != [entry->second length]) {
        throwJavaRuntimeException(env, "Zero buffer was not found or its size does not match");
        return;
    }
    memset([entry->second contents], 0, [entry->second length]);
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_nativeZeroBuffer(
        JNIEnv* env, jclass, jint bufferId, jint floatCount) {
    @autoreleasepool { zeroTypedBuffer(env, bufferId, floatCount, 1); }
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_nativeZeroTypedBuffer(
        JNIEnv* env, jclass, jint bufferId, jint elements, jint dtype) {
    @autoreleasepool { zeroTypedBuffer(env, bufferId, elements, dtype); }
}

static void downloadBuffer(JNIEnv* env, jint bufId, jfloatArray outArr) {
    std::lock_guard<std::mutex> lock(gBufferMutex);
    BufferEntry entry = transferEntry(env, bufId, outArr, sizeof(float), "Download");
    if (entry == gBufferPool.end()) return;
    jint len = env->GetArrayLength(outArr);
    env->SetFloatArrayRegion(outArr, 0, len, (const jfloat*)[entry->second contents]);
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
    for (int i = 0; i < count; i++) {
        gBufferPool.erase(ids[i]);
        gBufferDTypes.erase(ids[i]);
    }
    env->ReleaseIntArrayElements(idsArr, ids, JNI_ABORT);
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_nativeReleaseBuffers(
        JNIEnv* env, jclass, jintArray idsArr, jint count) {
    @autoreleasepool { releaseBuffers(env, idsArr, count); }
}

struct GraphLayoutWords {
    const jlong* words;
    int count;
    id<MTLBuffer> buffer;
};

struct GraphState {
    MetalContext* ctx;
    id<MTLCommandBuffer> commandBuffer;
    id<MTLComputeCommandEncoder> encoder;
    GraphLayoutWords layouts;

    ~GraphState() {
        if (encoder != nil) [encoder endEncoding];
    }
};

static id<MTLCommandBuffer> graphCommands(id<MTLCommandQueue> queue) {
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    if (commands == nil) throw std::runtime_error("Failed to create Metal command buffer");
    return commands;
}

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

static std::array<uint32_t, 4> storageCopyRange(const jint* cmd, int pos,
                                              id<MTLBuffer> input, id<MTLBuffer> output) {
    int dtype = gBufferDTypes.at(cmd[pos + 1]);
    if (cmd[pos + 1] == cmd[pos + 2] || dtype != gBufferDTypes.at(cmd[pos + 2])) {
        throw std::runtime_error("Raw GPU copies require matching dtypes and distinct allocations");
    }
    uint32_t count = (uint32_t)positiveCount(cmd[pos + 5], "copy elements");
    uint32_t width = dtype == 1 ? sizeof(float) : sizeof(jshort);
    if (cmd[pos + 3] < 0 || cmd[pos + 4] < 0 ||
        ((NSUInteger)cmd[pos + 3] + count) * width > [input length] ||
        ((NSUInteger)cmd[pos + 4] + count) * width > [output length]) {
        throw std::runtime_error("GPU copy range exceeds its physical allocation");
    }
    return {count, (uint32_t)cmd[pos + 3], (uint32_t)cmd[pos + 4], width};
}

static void encodeStorageCopy(GraphState& state, const jint* cmd, int pos) {
    id<MTLBuffer> input = requireStorageBuffer(cmd[pos + 1], "storage copy");
    id<MTLBuffer> output = requireStorageBuffer(cmd[pos + 2], "storage copy");
    auto range = storageCopyRange(cmd, pos, input, output);
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:state.ctx->storageCopyPSO];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    bindValue(encoder, range, 2);
    dispatch1D(encoder, state.ctx->storageCopyPSO, range[0]);
}

struct LogicalLayout {
    const jlong* axes;
    uint32_t rank, count, storage, offset;
};

static uint32_t layoutWord(jlong value) {
    if (value < 0 || value > INT32_MAX) throw std::runtime_error("Invalid GPU layout word");
    return (uint32_t)value;
}

static LogicalLayout layoutHeader(const GraphLayoutWords& stream, int start) {
    if (stream.words == nullptr || start < 0 || start > stream.count - 4) {
        throw std::runtime_error("Missing or truncated GPU layout descriptor");
    }
    const jlong* head = stream.words + start;
    uint32_t rank = layoutWord(head[0]);
    if (rank > (uint32_t)(stream.count - start - 4) / 2) {
        throw std::runtime_error("Truncated GPU layout axes");
    }
    LogicalLayout result{head + 4, rank, layoutWord(head[1]), layoutWord(head[2]), layoutWord(head[3])};
    if (result.count == 0 || result.offset >= result.storage) throw std::runtime_error("Empty GPU layout dispatch");
    return result;
}

static uint64_t layoutAxisExtent(const LogicalLayout& layout, uint32_t axis, uint64_t last) {
    uint32_t dimension = layoutWord(layout.axes[axis]);
    jlong stride = layout.axes[layout.rank + axis];
    if (dimension == 0 || stride < 0) throw std::runtime_error("Invalid GPU layout dimension or stride");
    uint64_t steps = dimension - 1;
    if (steps != 0 && (uint64_t)stride > (layout.storage - 1 - last) / steps) {
        throw std::runtime_error("GPU layout exceeds its physical storage");
    }
    return last + steps * (uint64_t)stride;
}

static void validateLayoutAxes(const LogicalLayout& layout) {
    uint64_t product = 1, last = layout.offset;
    for (uint32_t axis = 0; axis < layout.rank; axis++) {
        uint32_t dimension = layoutWord(layout.axes[axis]);
        if (dimension == 0 || product > layout.count / dimension) {
            throw std::runtime_error("GPU layout element count does not match dimensions");
        }
        product *= dimension;
        last = layoutAxisExtent(layout, axis, last);
    }
    if (product != layout.count) throw std::runtime_error("GPU layout count does not match its shape");
}

static LogicalLayout requireLogicalLayout(const GraphLayoutWords& stream, int start,
                                          id<MTLBuffer> buffer, int dtype) {
    LogicalLayout result = layoutHeader(stream, start);
    if (storageBytes((int)result.storage, dtype) != [buffer length]) {
        throw std::runtime_error("GPU layout storage does not match its buffer");
    }
    validateLayoutAxes(result);
    return result;
}

static id<MTLBuffer> layoutGpuBuffer(GraphState& state) {
    if (state.layouts.buffer != nil) return state.layouts.buffer;
    state.layouts.buffer = [state.ctx->device newBufferWithBytes:state.layouts.words
            length:(NSUInteger)state.layouts.count * sizeof(jlong) options:MTLResourceStorageModeShared];
    if (state.layouts.buffer == nil) throw std::runtime_error("Failed to allocate GPU layout metadata");
    return state.layouts.buffer;
}

static std::array<uint32_t, 4> layoutTransferParams(GraphState& state, const jint* cmd, int pos,
                                                   id<MTLBuffer> input, id<MTLBuffer> output) {
    if (cmd[pos + 1] == cmd[pos + 2]) throw std::runtime_error("Layout transfer requires distinct allocations");
    int sourceType = gBufferDTypes.at(cmd[pos + 1]), targetType = gBufferDTypes.at(cmd[pos + 2]);
    auto layout = requireLogicalLayout(state.layouts, cmd[pos + 3], input, sourceType);
    if (storageBytes((int)layout.count, targetType) != [output length]) {
        throw std::runtime_error("Layout transfer output must match logical element count");
    }
    return {(uint32_t)cmd[pos + 3], (uint32_t)sourceType, (uint32_t)targetType, layout.count};
}

template <typename T>
static void encodeLayoutPair(GraphState& state, id<MTLComputePipelineState> pipeline,
                             id<MTLBuffer> input, id<MTLBuffer> output, const T& params, uint32_t count) {
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBuffer:layoutGpuBuffer(state) offset:0 atIndex:2];
    bindValue(encoder, params, 3);
    dispatch1D(encoder, pipeline, count);
}

static void encodeLayoutTransfer(GraphState& state, const jint* cmd, int pos) {
    id<MTLBuffer> input = requireStorageBuffer(cmd[pos + 1], "layout transfer");
    id<MTLBuffer> output = requireStorageBuffer(cmd[pos + 2], "layout transfer");
    auto params = layoutTransferParams(state, cmd, pos, input, output);
    encodeLayoutPair(state, state.ctx->layoutTransferPSO, input, output, params, params[3]);
}

static void requireSameLayoutShape(const LogicalLayout& first, const LogicalLayout& second) {
    if (first.rank != second.rank || first.count != second.count ||
        !std::equal(first.axes, first.axes + first.rank, second.axes)) {
        throw std::runtime_error("GPU operand layout shapes must match");
    }
}

static void requireWritableLayout(const LogicalLayout& layout) {
    std::vector<std::pair<uint64_t, uint32_t>> axes;
    for (uint32_t axis = 0; axis < layout.rank; axis++) {
        if (layout.axes[axis] > 1) axes.emplace_back(layout.axes[layout.rank + axis], layout.axes[axis]);
    }
    std::sort(axes.begin(), axes.end());
    uint64_t span = 1;
    for (auto [stride, dimension] : axes) {
        if (stride < span) throw std::runtime_error("Overlapping GPU output layouts cannot be written");
        span += (dimension - 1) * stride;
    }
}

static std::array<uint32_t, 5> layoutAssignParams(GraphState& state, const jint* cmd, int pos,
                                                id<MTLBuffer> input, id<MTLBuffer> output) {
    if (cmd[pos + 1] == cmd[pos + 2]) throw std::runtime_error("GPU assignment requires an alias snapshot");
    int sourceType = gBufferDTypes.at(cmd[pos + 1]), targetType = gBufferDTypes.at(cmd[pos + 2]);
    auto source = requireLogicalLayout(state.layouts, cmd[pos + 3], input, sourceType);
    auto target = requireLogicalLayout(state.layouts, cmd[pos + 4], output, targetType);
    requireSameLayoutShape(source, target);
    requireWritableLayout(target);
    return {(uint32_t)cmd[pos + 3], (uint32_t)cmd[pos + 4], (uint32_t)sourceType,
            (uint32_t)targetType, source.count};
}

static void encodeLayoutAssign(GraphState& state, const jint* cmd, int pos) {
    id<MTLBuffer> input = requireStorageBuffer(cmd[pos + 1], "layout assign");
    id<MTLBuffer> output = requireStorageBuffer(cmd[pos + 2], "layout assign");
    auto params = layoutAssignParams(state, cmd, pos, input, output);
    encodeLayoutPair(state, state.ctx->layoutAssignPSO, input, output, params, params[4]);
}

static void requireLayoutBinaryOperation(int operation) {
    if (operation >= OP_ADD && operation <= OP_DIVIDE) return;
    if (operation == OP_RELU_BACKWARD || operation == OP_GELU_BACKWARD) return;
    throw std::runtime_error("Invalid GPU layout binary operation");
}

static void requireLayoutUnaryOperation(int operation) {
    if (operation >= OP_MULTIPLY_SCALAR && operation <= OP_RELU) return;
    if (operation == OP_GELU || operation == OP_ADD_SCALAR || operation == OP_DIVIDE_SCALAR ||
        operation == OP_CLAMP || operation == OP_POW) return;
    throw std::runtime_error("Invalid GPU layout unary operation");
}

static std::array<uint32_t, 6> layoutUnaryParams(GraphState& state, const jint* cmd, int pos,
                                               id<MTLBuffer> input, id<MTLBuffer> output) {
    requireLayoutUnaryOperation(cmd[pos + 4]);
    auto transfer = layoutTransferParams(state, cmd, pos, input, output);
    if (transfer[1] != transfer[2]) throw std::runtime_error("GPU unary output must retain input dtype");
    return {transfer[0], transfer[1], (uint32_t)cmd[pos + 4], (uint32_t)cmd[pos + 5],
            (uint32_t)cmd[pos + 6], transfer[3]};
}

static void encodeLayoutUnary(GraphState& state, const jint* cmd, int pos) {
    id<MTLBuffer> input = requireStorageBuffer(cmd[pos + 1], "layout unary");
    id<MTLBuffer> output = requireStorageBuffer(cmd[pos + 2], "layout unary");
    auto params = layoutUnaryParams(state, cmd, pos, input, output);
    encodeLayoutPair(state, state.ctx->layoutUnaryPSO, input, output, params, params[5]);
}

static std::array<uint32_t, 7> layoutBinaryParams(GraphState& state, const jint* cmd, int pos) {
    requireLayoutBinaryOperation(cmd[pos + 6]);
    int a = cmd[pos + 1], b = cmd[pos + 2], out = cmd[pos + 3];
    if (a == out || b == out) throw std::runtime_error("GPU binary output must be independent");
    int at = gBufferDTypes.at(a), bt = gBufferDTypes.at(b), ot = gBufferDTypes.at(out);
    auto first = requireLogicalLayout(state.layouts, cmd[pos + 4], requireStorageBuffer(a, "binary"), at);
    auto second = requireLogicalLayout(state.layouts, cmd[pos + 5], requireStorageBuffer(b, "binary"), bt);
    requireSameLayoutShape(first, second);
    if (ot != (at == bt ? at : 1) || storageBytes((int)first.count, ot) !=
        [requireStorageBuffer(out, "binary output") length]) throw std::runtime_error("Invalid GPU binary output");
    return {(uint32_t)cmd[pos + 4], (uint32_t)cmd[pos + 5], (uint32_t)at, (uint32_t)bt,
            (uint32_t)ot, (uint32_t)cmd[pos + 6], first.count};
}

static void encodeLayoutBinary(GraphState& state, const jint* cmd, int pos) {
    auto params = layoutBinaryParams(state, cmd, pos);
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:state.ctx->layoutBinaryPSO];
    [encoder setBuffer:requireStorageBuffer(cmd[pos + 1], "binary") offset:0 atIndex:0];
    [encoder setBuffer:requireStorageBuffer(cmd[pos + 2], "binary") offset:0 atIndex:1];
    [encoder setBuffer:requireStorageBuffer(cmd[pos + 3], "binary") offset:0 atIndex:2];
    [encoder setBuffer:layoutGpuBuffer(state) offset:0 atIndex:3];
    bindValue(encoder, params, 4);
    dispatch1D(encoder, state.ctx->layoutBinaryPSO, params[6]);
}

struct GroupedLayout {
    LogicalLayout layout;
    uint32_t groups, width;
};

static GroupedLayout requireGroupedLayout(GraphState& state, int start, int retained, int buffer) {
    auto layout = requireLogicalLayout(state.layouts, start, requireStorageBuffer(buffer, "grouped layout"),
                                       gBufferDTypes.at(buffer));
    if (retained < 0 || (uint32_t)retained > layout.rank) throw std::runtime_error("Invalid retained GPU axes");
    uint32_t groups = 1;
    for (int axis = 0; axis < retained; axis++) {
        groups *= (uint32_t)layout.axes[axis];
    }
    return {layout, groups, layout.count / groups};
}

static void requireReductionOperation(int operation, int correction) {
    if (operation < 0 || operation > 5) throw std::runtime_error("Invalid GPU reduction operation");
    if (correction < 0 || (operation != 2 && correction != 0)) {
        throw std::runtime_error("Invalid GPU variance correction");
    }
}

static std::array<uint32_t, 7> layoutReduceParams(GraphState& state, const jint* cmd, int pos) {
    requireReductionOperation(cmd[pos + 5], cmd[pos + 6]);
    int source = cmd[pos + 1], target = cmd[pos + 2];
    if (source == target) throw std::runtime_error("Reduction output requires independent storage");
    auto group = requireGroupedLayout(state, cmd[pos + 3], cmd[pos + 4], source);
    int sourceType = gBufferDTypes.at(source), targetType = gBufferDTypes.at(target);
    if ((targetType != sourceType && targetType != 1) || (cmd[pos + 5] >= 4 && targetType != 1) ||
        storageBytes((int)group.groups, targetType) != [requireStorageBuffer(target, "reduce output") length]) {
        throw std::runtime_error("Invalid grouped GPU reduction output");
    }
    return {(uint32_t)cmd[pos + 3], (uint32_t)sourceType, (uint32_t)targetType, group.groups,
            group.width, (uint32_t)cmd[pos + 5], (uint32_t)cmd[pos + 6]};
}

static void encodeLayoutReduce(GraphState& state, const jint* cmd, int pos) {
    auto params = layoutReduceParams(state, cmd, pos);
    encodeLayoutPair(state, state.ctx->layoutReducePSO, requireStorageBuffer(cmd[pos + 1], "reduce"),
                     requireStorageBuffer(cmd[pos + 2], "reduce"), params, params[3]);
}

static LogicalLayout requireCompleteOutput(GraphState& state, int start, int target,
                                           const LogicalLayout& input, int dtype) {
    auto output = requireLogicalLayout(state.layouts, start, requireStorageBuffer(target, "layout output"),
                                       gBufferDTypes.at(target));
    requireSameLayoutShape(input, output);
    requireWritableLayout(output);
    if (gBufferDTypes.at(target) != dtype || output.count != output.storage) {
        throw std::runtime_error("Invalid complete GPU output layout");
    }
    return output;
}

static std::array<uint32_t, 5> layoutSoftmaxParams(GraphState& state, const jint* cmd, int pos) {
    int source = cmd[pos + 1], target = cmd[pos + 2];
    if (source == target) throw std::runtime_error("Softmax output requires independent storage");
    auto group = requireGroupedLayout(state, cmd[pos + 3], cmd[pos + 5], source);
    if ((uint32_t)cmd[pos + 5] + 1 != group.layout.rank) throw std::runtime_error("Softmax requires one axis");
    int dtype = gBufferDTypes.at(source);
    requireCompleteOutput(state, cmd[pos + 4], target, group.layout, dtype);
    return {(uint32_t)cmd[pos + 3], (uint32_t)cmd[pos + 4], (uint32_t)dtype, group.width, group.groups};
}

static void encodeLayoutSoftmax(GraphState& state, const jint* cmd, int pos) {
    auto params = layoutSoftmaxParams(state, cmd, pos);
    encodeLayoutPair(state, state.ctx->layoutSoftmaxPSO, requireStorageBuffer(cmd[pos + 1], "softmax"),
                     requireStorageBuffer(cmd[pos + 2], "softmax"), params, params[4]);
}

static void requireMaximumUpstream(GraphState& state, const jint* cmd, int pos, uint32_t groups, int dtype) {
    int maximum = cmd[pos + 2], gradient = cmd[pos + 3];
    auto winner = requireLogicalLayout(state.layouts, cmd[pos + 6], requireStorageBuffer(maximum, "maximum"),
                                       gBufferDTypes.at(maximum));
    auto upstream = requireLogicalLayout(state.layouts, cmd[pos + 7], requireStorageBuffer(gradient, "gradient"),
                                         gBufferDTypes.at(gradient));
    requireSameLayoutShape(winner, upstream);
    if (winner.count != groups || gBufferDTypes.at(maximum) != dtype || gBufferDTypes.at(gradient) != 1) {
        throw std::runtime_error("Invalid maximum adjoint values or dtypes");
    }
}

static std::array<uint32_t, 8> layoutMaximumParams(GraphState& state, const jint* cmd, int pos) {
    int target = cmd[pos + 4];
    if (target == cmd[pos + 1] || target == cmd[pos + 2] || target == cmd[pos + 3]) {
        throw std::runtime_error("Maximum adjoint output requires independent storage");
    }
    auto group = requireGroupedLayout(state, cmd[pos + 5], cmd[pos + 9], cmd[pos + 1]);
    int dtype = gBufferDTypes.at(cmd[pos + 1]);
    requireMaximumUpstream(state, cmd, pos, group.groups, dtype);
    requireCompleteOutput(state, cmd[pos + 8], target, group.layout, 1);
    return {(uint32_t)cmd[pos + 5], (uint32_t)cmd[pos + 6], (uint32_t)cmd[pos + 7],
            (uint32_t)cmd[pos + 8], (uint32_t)dtype, (uint32_t)dtype, group.width, group.groups};
}

static void encodeLayoutMaximumBackward(GraphState& state, const jint* cmd, int pos) {
    auto params = layoutMaximumParams(state, cmd, pos);
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:state.ctx->layoutMaximumBackwardPSO];
    for (NSUInteger index = 0; index < 4; index++) {
        [encoder setBuffer:requireStorageBuffer(cmd[pos + 1 + index], "maximum adjoint") offset:0 atIndex:index];
    }
    [encoder setBuffer:layoutGpuBuffer(state) offset:0 atIndex:4];
    bindValue(encoder, params, 5);
    dispatch1D(encoder, state.ctx->layoutMaximumBackwardPSO, params[7]);
}

struct IndexPlan {
    uint32_t count, groups;
    const jlong* indices;
    const jlong* targets;
    const jlong* offsets;
    const jlong* positions;
};

static IndexPlan indexPlanHeader(const GraphLayoutWords& stream, int start) {
    if (stream.words == nullptr || start < 0 || start > stream.count - 2) {
        throw std::runtime_error("Missing GPU index metadata");
    }
    const jlong* words = stream.words + start;
    uint32_t count = layoutWord(words[0]), groups = layoutWord(words[1]);
    if (count == 0 || groups == 0 || groups > count ||
        3 + 2ULL * count + 2ULL * groups > (uint64_t)(stream.count - start)) {
        throw std::runtime_error("Invalid or truncated GPU index metadata");
    }
    const jlong* indices = words + 2;
    return {count, groups, indices, indices + count, indices + count + groups,
            indices + count + 2 * groups + 1};
}

static void validateIndexGroup(const IndexPlan& plan, uint32_t group, std::vector<bool>& seen) {
    uint32_t first = layoutWord(plan.offsets[group]), end = layoutWord(plan.offsets[group + 1]);
    if (first >= end || end > plan.count) throw std::runtime_error("Invalid GPU index group bounds");
    int64_t previous = -1;
    for (uint32_t position = first; position < end; position++) {
        uint32_t source = layoutWord(plan.positions[position]);
        if (source >= plan.count || seen[source] || source <= previous ||
            plan.indices[source] != plan.targets[group]) {
            throw std::runtime_error("GPU index groups must preserve each source position once in order");
        }
        seen[source] = true;
        previous = source;
    }
}

static IndexPlan requireIndexPlan(const GraphLayoutWords& stream, int start, uint32_t width) {
    auto plan = indexPlanHeader(stream, start);
    for (uint32_t index = 0; index < plan.count; index++) {
        if (layoutWord(plan.indices[index]) >= width) throw std::runtime_error("GPU index out of range");
    }
    std::vector<jlong> targets(plan.targets, plan.targets + plan.groups);
    std::sort(targets.begin(), targets.end());
    if (targets.front() < 0 || targets.back() >= width ||
        std::adjacent_find(targets.begin(), targets.end()) != targets.end() ||
        plan.offsets[0] != 0 || plan.offsets[plan.groups] != plan.count) {
        throw std::runtime_error("Invalid GPU index destinations or coverage");
    }
    std::vector<bool> seen(plan.count, false);
    for (uint32_t group = 0; group < plan.groups; group++) {
        validateIndexGroup(plan, group, seen);
    }
    return plan;
}

static uint32_t indexTrailing(const LogicalLayout& layout, int axis) {
    if (axis < 0 || (uint32_t)axis >= layout.rank) throw std::runtime_error("Invalid GPU index axis");
    uint32_t trailing = 1;
    for (uint32_t next = axis + 1; next < layout.rank; next++) {
        trailing *= layout.axes[next];
    }
    return trailing;
}

static std::array<uint32_t, 7> layoutIndexSelectParams(GraphState& state, const jint* cmd, int pos) {
    int input = cmd[pos + 1], output = cmd[pos + 2], dtype = gBufferDTypes.at(input);
    auto source = requireLogicalLayout(state.layouts, cmd[pos + 3], requireStorageBuffer(input, "gather"), dtype);
    uint32_t trailing = indexTrailing(source, cmd[pos + 5]), width = source.axes[cmd[pos + 5]];
    auto plan = requireIndexPlan(state.layouts, cmd[pos + 4], width);
    uint64_t count = (uint64_t)(source.count / width) * plan.count;
    if (input == output || gBufferDTypes.at(output) != dtype || count > INT32_MAX ||
        [requireStorageBuffer(output, "gather output") length] != count * (dtype == 1 ? 4 : 2)) {
        throw std::runtime_error("Invalid typed GPU gather output");
    }
    return {(uint32_t)cmd[pos + 3], (uint32_t)cmd[pos + 4], width, trailing, plan.count,
            (uint32_t)dtype, (uint32_t)count};
}

static void encodeLayoutIndexSelect(GraphState& state, const jint* cmd, int pos) {
    auto params = layoutIndexSelectParams(state, cmd, pos);
    encodeLayoutPair(state, state.ctx->layoutIndexSelectPSO, requireStorageBuffer(cmd[pos + 1], "gather"),
                     requireStorageBuffer(cmd[pos + 2], "gather"), params, params[6]);
}

static void requireIndexUpdateShape(const LogicalLayout& target, const LogicalLayout& updates,
                                    uint32_t axis, uint32_t count) {
    if (target.rank != updates.rank) throw std::runtime_error("Indexed updates must match target rank");
    for (uint32_t dimension = 0; dimension < target.rank; dimension++) {
        uint32_t expected = dimension == axis ? count : target.axes[dimension];
        if (updates.axes[dimension] != expected) throw std::runtime_error("Invalid indexed update shape");
    }
    requireWritableLayout(target);
}

static std::array<uint32_t, 9> layoutIndexAddParams(GraphState& state, const jint* cmd, int pos) {
    int target = cmd[pos + 1], input = cmd[pos + 2], dtype = gBufferDTypes.at(target), from = gBufferDTypes.at(input);
    auto output = requireLogicalLayout(state.layouts, cmd[pos + 3], requireStorageBuffer(target, "scatter"), dtype);
    auto updates = requireLogicalLayout(state.layouts, cmd[pos + 4], requireStorageBuffer(input, "scatter"), from);
    uint32_t trailing = indexTrailing(output, cmd[pos + 6]), width = output.axes[cmd[pos + 6]];
    auto plan = requireIndexPlan(state.layouts, cmd[pos + 5], width);
    requireIndexUpdateShape(output, updates, cmd[pos + 6], plan.count);
    if (target == input || (dtype != 1 && from != dtype)) {
        throw std::runtime_error("Indexed updates require an alias snapshot and cannot narrow dtype");
    }
    return {(uint32_t)cmd[pos + 3], (uint32_t)cmd[pos + 4], (uint32_t)cmd[pos + 5], width, trailing,
            (uint32_t)dtype, (uint32_t)from, plan.groups, output.count / width * plan.groups};
}

static void encodeLayoutIndexAdd(GraphState& state, const jint* cmd, int pos) {
    auto params = layoutIndexAddParams(state, cmd, pos);
    encodeLayoutPair(state, state.ctx->layoutIndexAddPSO, requireStorageBuffer(cmd[pos + 1], "scatter"),
                     requireStorageBuffer(cmd[pos + 2], "scatter"), params, params[8]);
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
    id<MTLComputePipelineState> pipeline = scalarPipeline(state.ctx, cmd[pos]);
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "scalar op") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "scalar op") offset:0 atIndex:1];
    bindValue(encoder, scalar, 2);
    dispatch1D(encoder, pipeline, positiveCount(cmd[pos + 4], "element count"));
}

static void encodeClamp(GraphState& state, const jint* cmd, int pos) {
    float range[] = {floatFromBits(cmd[pos + 3]), floatFromBits(cmd[pos + 4])};
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->clampPSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "clamp op") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "clamp op") offset:0 atIndex:1];
    bindValue(encoder, range, 2);
    dispatch1D(encoder, state.ctx->clampPSO,
               positiveCount(cmd[pos + 5], "element count"));
}

static void encodeTranspose(GraphState& state, const jint* cmd, int pos) {
    uint32_t dims[] = {(uint32_t)cmd[pos + 3], (uint32_t)cmd[pos + 4]};
    NSUInteger total = elementCount(cmd[pos + 3], cmd[pos + 4]);
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->transposePSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "transpose") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "transpose") offset:0 atIndex:1];
    bindValue(encoder, dims, 2);
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
    id<MTLComputePipelineState> pipeline = headPermutationPipeline(state.ctx, cmd[pos]);
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "head permutation") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "head permutation") offset:0 atIndex:1];
    bindValue(encoder, dims, 2);
    dispatch1D(encoder, pipeline, total);
}

static void encodeCausalMask(GraphState& state, const jint* cmd, int pos) {
    uint32_t dims[] = {(uint32_t)cmd[pos + 3], (uint32_t)cmd[pos + 4]};
    NSUInteger total = elementCount(cmd[pos + 3], cmd[pos + 4]);
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->causalMaskPSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "causal mask") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "causal mask") offset:0 atIndex:1];
    bindValue(encoder, dims, 2);
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
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->causalSoftmaxPSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "causal softmax") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "causal softmax") offset:0 atIndex:1];
    bindValue(encoder, params, 2);
    NSUInteger threads = rowReductionWidth(state.ctx->causalSoftmaxPSO);
    [encoder dispatchThreads:MTLSizeMake(threads, params.rows, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static void encodeRotary(GraphState& state, const jint* cmd, int pos) {
    uint32_t dims[] = {(uint32_t)cmd[pos + 5], (uint32_t)cmd[pos + 6],
                       (uint32_t)cmd[pos + 7], (uint32_t)cmd[pos + 8]};
    NSUInteger total = elementCount(cmd[pos + 5], cmd[pos + 7] / 2);
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->rotaryPSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "rotary input") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "rotary cosine") offset:0 atIndex:1];
    [encoder setBuffer:requireBuffer(cmd[pos + 3], "rotary sine") offset:0 atIndex:2];
    [encoder setBuffer:requireBuffer(cmd[pos + 4], "rotary output") offset:0 atIndex:3];
    bindValue(encoder, dims, 4);
    dispatch1D(encoder, state.ctx->rotaryPSO, total);
}

static id<MTLComputePipelineState> scatterPipeline(MetalContext* ctx, int op) {
    if (op == OP_SCATTER_ADD_ROWS) return ctx->scatterAddRowsPSO;
    if (op == OP_SCATTER_ADD_ROWS_ATOMIC) return ctx->orderedScatterPSO;
    throw std::runtime_error("Invalid scatter op");
}

static id<MTLBuffer> encodeScatterLinks(GraphState& state, const jint* cmd,
                                       int pos, const std::array<uint32_t, 3>& dimensions) {
    if (cmd[pos] != OP_SCATTER_ADD_ROWS_ATOMIC) return nil;
    NSUInteger count = positiveCount(cmd[pos + 6], "scatter index count");
    id<MTLBuffer> links = [state.ctx->device newBufferWithLength:count * sizeof(uint32_t) * 2
                                                     options:MTLResourceStorageModeShared];
    if (links == nil) throw std::runtime_error("Failed to allocate scatter links");
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->scatterLinksPSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "scatter indices") offset:0 atIndex:0];
    [encoder setBuffer:links offset:0 atIndex:1];
    bindValue(encoder, dimensions, 2);
    dispatch1D(encoder, state.ctx->scatterLinksPSO, count);
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    return links;
}

static void encodeScatter(GraphState& state, const jint* cmd, int pos) {
    std::array<uint32_t, 3> dims{(uint32_t)cmd[pos + 4], (uint32_t)cmd[pos + 5],
                                (uint32_t)cmd[pos + 6]};
    NSUInteger total = elementCount(cmd[pos + 5], cmd[pos + 6]);
    id<MTLBuffer> links = encodeScatterLinks(state, cmd, pos, dims);
    id<MTLComputePipelineState> pipeline = scatterPipeline(state.ctx, cmd[pos]);
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "scatter") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "scatter") offset:0 atIndex:1];
    [encoder setBuffer:requireBuffer(cmd[pos + 3], "scatter") offset:0 atIndex:2];
    bindValue(encoder, dims, 3);
    if (links != nil) [encoder setBuffer:links offset:0 atIndex:4];
    dispatch1D(encoder, pipeline, total);
}

static void encodeGatherRows(GraphState& state, const jint* cmd, int pos) {
    uint32_t dims[] = {(uint32_t)cmd[pos + 4], (uint32_t)cmd[pos + 5],
                       (uint32_t)cmd[pos + 6]};
    NSUInteger total = elementCount(cmd[pos + 5], cmd[pos + 6]);
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->gatherRowsPSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "gather input") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "gather indices") offset:0 atIndex:1];
    [encoder setBuffer:requireBuffer(cmd[pos + 3], "gather output") offset:0 atIndex:2];
    bindValue(encoder, dims, 3);
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
    id<MTLComputePipelineState> pipeline = broadcastPipeline(state.ctx, cmd[pos]);
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "broadcast") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "broadcast") offset:0 atIndex:1];
    [encoder setBuffer:requireBuffer(cmd[pos + 3], "broadcast") offset:0 atIndex:2];
    bindValue(encoder, dims, 3);
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
    id<MTLComputePipelineState> pipeline = reductionPipeline(state.ctx, cmd[pos]);
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "reduction") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "reduction") offset:0 atIndex:1];
    bindValue(encoder, dims, 2);
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
    id<MTLComputePipelineState> pipeline = rowStatisticPipeline(state.ctx, cmd[pos]);
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "row statistic") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "row statistic") offset:0 atIndex:1];
    bindValue(encoder, dims, 2);
    NSUInteger threads = MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(threads, rows, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static void encodeSquaresTotal(GraphState& state, id<MTLBuffer> partials,
                                id<MTLBuffer> total, NSUInteger groups) {
    uint32_t count = (uint32_t)groups;
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:state.ctx->accumulateSquaresPSO];
    [encoder setBuffer:partials offset:0 atIndex:0];
    [encoder setBuffer:total offset:0 atIndex:1];
    bindValue(encoder, count, 2);
    NSUInteger threads = MIN((NSUInteger)256,
                             state.ctx->accumulateSquaresPSO.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

static void encodeSumSquaresScalar(GraphState& state, const jint* cmd, int pos) {
    NSUInteger count = positiveCount(cmd[pos + 3], "sum squares count");
    NSUInteger threads = MIN((NSUInteger)256,
                             state.ctx->sumSquaresScalarPSO.maxTotalThreadsPerThreadgroup);
    NSUInteger groups = MIN((count + threads - 1) / threads, (NSUInteger)1024);
    uint32_t dims[] = {(uint32_t)count, (uint32_t)groups};
    id<MTLBuffer> partials = [state.ctx->device newBufferWithLength:groups * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
    if (partials == nil) throw std::runtime_error("Failed to allocate norm partials");
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->sumSquaresScalarPSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "sum squares") offset:0 atIndex:0];
    [encoder setBuffer:partials offset:0 atIndex:1];
    bindValue(encoder, dims, 2);
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    encodeSquaresTotal(state, partials, requireBuffer(cmd[pos + 2], "sum squares"), groups);
}

static void encodeColumnSum(GraphState& state, id<MTLBuffer> input,
                            id<MTLBuffer> output, const std::array<uint32_t, 2>& dims, NSUInteger cols) {
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->sumRowsPSO];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    bindValue(encoder, dims, 2);
    NSUInteger threads = colReductionHeight(state.ctx->sumRowsPSO);
    [encoder dispatchThreads:MTLSizeMake(cols, threads, 1)
       threadsPerThreadgroup:MTLSizeMake(1, threads, 1)];
}

static void encodeFinalSum(GraphState& state, id<MTLBuffer> input,
                           id<MTLBuffer> output, const std::array<uint32_t, 2>& dims) {
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:state.ctx->sumAlongRowsPSO];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    bindValue(encoder, dims, 2);
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
}

static void encodeScalarSum(GraphState& state, const jint* cmd, int pos) {
    NSUInteger rows = positiveCount(cmd[pos + 3], "rows");
    NSUInteger cols = positiveCount(cmd[pos + 4], "cols");
    std::array<uint32_t, 2> inputDims{(uint32_t)rows, (uint32_t)cols};
    std::array<uint32_t, 2> outputDims{1u, (uint32_t)cols};
    id<MTLBuffer> temporary = [state.ctx->device
        newBufferWithLength:cols * sizeof(float) options:MTLResourceStorageModeShared];
    if (temporary == nil) throw std::runtime_error("Failed to allocate sum buffer");
    encodeColumnSum(state, requireBuffer(cmd[pos + 1], "sum"), temporary,
                    inputDims, cols);
    encodeFinalSum(state, temporary, requireBuffer(cmd[pos + 2], "sum"),
                   outputDims);
}

static id<MTLComputePipelineState> crossEntropyPipeline(MetalContext* ctx, int op) {
    if (op == OP_CROSS_ENTROPY_LOSS) return ctx->crossEntropyLossPSO;
    if (op == OP_CROSS_ENTROPY_GRADIENT) return ctx->crossEntropyGradPSO;
    throw std::runtime_error("Invalid cross entropy op");
}

static void encodeCrossEntropy(GraphState& state, const jint* cmd, int pos) {
    NSUInteger rows = positiveCount(cmd[pos + 4], "rows");
    uint32_t dims[] = {(uint32_t)rows, (uint32_t)positiveCount(cmd[pos + 5], "cols")};
    id<MTLComputePipelineState> pipeline = crossEntropyPipeline(state.ctx, cmd[pos]);
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "cross entropy") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "cross entropy") offset:0 atIndex:1];
    [encoder setBuffer:requireBuffer(cmd[pos + 3], "cross entropy") offset:0 atIndex:2];
    bindValue(encoder, dims, 3);
    NSUInteger threads = rowReductionWidth(pipeline);
    [encoder dispatchThreads:MTLSizeMake(threads, rows, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static void encodeFusedCrossEntropy(GraphState& state, const jint* cmd, int pos) {
    NSUInteger rows = positiveCount(cmd[pos + 5], "rows");
    uint32_t dims[] = {(uint32_t)rows, (uint32_t)positiveCount(cmd[pos + 6], "cols")};
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->crossEntropyFusedPSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "cross entropy") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "cross entropy") offset:0 atIndex:1];
    [encoder setBuffer:requireBuffer(cmd[pos + 3], "cross entropy") offset:0 atIndex:2];
    [encoder setBuffer:requireBuffer(cmd[pos + 4], "cross entropy") offset:0 atIndex:3];
    bindValue(encoder, dims, 4);
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

struct LayoutMatmulSpec {
    LogicalLayout left, right;
    uint32_t rows, columns, inner, batches, count, leftType, rightType, outputType;
};

static void requireMatmulLayouts(const LogicalLayout& left, const LogicalLayout& right) {
    if (left.rank < 2 || left.rank != right.rank ||
        !std::equal(left.axes, left.axes + left.rank - 2, right.axes) ||
        left.axes[left.rank - 1] != right.axes[right.rank - 2]) {
        throw std::runtime_error("Invalid promoted GPU matmul dimensions");
    }
}

static LayoutMatmulSpec layoutMatmulSpec(GraphState& state, const jint* cmd, int pos) {
    uint32_t a = gBufferDTypes.at(cmd[pos + 1]), b = gBufferDTypes.at(cmd[pos + 2]);
    auto left = requireLogicalLayout(state.layouts, cmd[pos + 4],
                                     requireStorageBuffer(cmd[pos + 1], "matmul left"), a);
    auto right = requireLogicalLayout(state.layouts, cmd[pos + 5],
                                      requireStorageBuffer(cmd[pos + 2], "matmul right"), b);
    requireMatmulLayouts(left, right);
    uint32_t m = left.axes[left.rank - 2], k = left.axes[left.rank - 1], n = right.axes[right.rank - 1];
    uint32_t batches = left.count / m / k, dtype = a == b ? a : 1;
    uint64_t count = (uint64_t)batches * m * n;
    id<MTLBuffer> output = requireStorageBuffer(cmd[pos + 3], "matmul output");
    if (count > INT32_MAX || output.length != count * (dtype == 1 ? 4 : 2) ||
        gBufferDTypes.at(cmd[pos + 3]) != (int)dtype || cmd[pos + 3] == cmd[pos + 1] ||
        cmd[pos + 3] == cmd[pos + 2]) throw std::runtime_error("Invalid GPU matmul output storage");
    return {left, right, m, n, k, batches, (uint32_t)count, a, b, dtype};
}

struct MpsLayoutMatrix {
    uint32_t rows, columns;
    uint64_t rowStride;
    bool transposed;
};

static bool mpsLayoutMatrix(const LogicalLayout& layout, MpsLayoutMatrix& matrix) {
    uint32_t rows = layout.axes[layout.rank - 2], cols = layout.axes[layout.rank - 1];
    uint64_t rowStride = layout.axes[2 * layout.rank - 2], colStride = layout.axes[2 * layout.rank - 1];
    if ((cols == 1 || colStride == 1) && (rows == 1 || rowStride >= cols)) {
        matrix = {rows, cols, rows == 1 ? cols : rowStride, false};
        return true;
    }
    if ((rows == 1 || rowStride == 1) && (cols == 1 || colStride >= rows)) {
        matrix = {cols, rows, cols == 1 ? rows : colStride, true};
        return true;
    }
    return false;
}

static NSUInteger matmulBatchAddress(const LogicalLayout& layout, uint32_t batch) {
    NSUInteger address = layout.offset;
    for (int axis = (int)layout.rank - 3; axis >= 0; axis--) {
        address += (batch % layout.axes[axis]) * layout.axes[layout.rank + axis];
        batch /= layout.axes[axis];
    }
    return address;
}

static uint64_t matmulBatchStride(const LogicalLayout& layout) {
    if (layout.rank == 2) return 0;
    uint64_t stride = 0, expected = 0;
    for (int axis = (int)layout.rank - 3; axis >= 0; axis--) {
        if (layout.axes[axis] == 1) continue;
        uint64_t step = layout.axes[layout.rank + axis];
        if (stride == 0) {
            stride = step;
            expected = step;
        }
        if (expected == 0 || step != expected) return 0;
        expected *= layout.axes[axis];
    }
    return stride;
}

static MPSMatrix* layoutMpsMatrix(id<MTLBuffer> buffer, const MpsLayoutMatrix& matrix,
                                  NSUInteger offset, uint32_t batches, uint64_t batchStride) {
    auto descriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:matrix.rows columns:matrix.columns
        matrices:batches rowBytes:matrix.rowStride * sizeof(float)
        matrixBytes:batchStride * sizeof(float) dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:buffer offset:offset * sizeof(float) descriptor:descriptor];
}

static bool mpsBatchCompatible(const MpsLayoutMatrix& matrix, uint64_t stride) {
    return stride >= matrix.rows * matrix.rowStride && stride % matrix.rowStride == 0;
}

static void encodeLayoutMpsBatch(GraphState& state, const jint* cmd, int pos, const LayoutMatmulSpec& spec,
                                 const MpsLayoutMatrix& a, const MpsLayoutMatrix& b,
                                 uint32_t batch, uint32_t count, uint64_t firstStride, uint64_t secondStride) {
    auto left = layoutMpsMatrix(requireBuffer(cmd[pos + 1], "layout matmul"), a,
                                matmulBatchAddress(spec.left, batch), count, firstStride);
    auto right = layoutMpsMatrix(requireBuffer(cmd[pos + 2], "layout matmul"), b,
                                 matmulBatchAddress(spec.right, batch), count, secondStride);
    MpsLayoutMatrix result{spec.rows, spec.columns, spec.columns, false};
    auto output = layoutMpsMatrix(requireBuffer(cmd[pos + 3], "layout matmul"), result,
                                  (NSUInteger)batch * spec.rows * spec.columns, count,
                                  (uint64_t)spec.rows * spec.columns);
    [configuredMatmulKernel(state.ctx, spec.rows, spec.columns, spec.inner, a.transposed, b.transposed, count)
        encodeToCommandBuffer:state.commandBuffer leftMatrix:left rightMatrix:right resultMatrix:output];
}

static bool encodeLayoutMpsMatmul(GraphState& state, const jint* cmd, int pos, const LayoutMatmulSpec& spec) {
    MpsLayoutMatrix a, b;
    if (spec.leftType != 1 || spec.rightType != 1 ||
        !mpsLayoutMatrix(spec.left, a) || !mpsLayoutMatrix(spec.right, b)) return false;
    endGraphEncoder(state);
    uint64_t first = matmulBatchStride(spec.left), second = matmulBatchStride(spec.right);
    if (mpsBatchCompatible(a, first) && mpsBatchCompatible(b, second)) {
        encodeLayoutMpsBatch(state, cmd, pos, spec, a, b, 0, spec.batches, first, second);
    } else {
        for (uint32_t batch = 0; batch < spec.batches; batch++) {
            encodeLayoutMpsBatch(state, cmd, pos, spec, a, b, batch, 1,
                                 a.rows * a.rowStride, b.rows * b.rowStride);
        }
    }
    return true;
}

static void encodeLayoutMatmul(GraphState& state, const jint* cmd, int pos) {
    auto spec = layoutMatmulSpec(state, cmd, pos);
    if (encodeLayoutMpsMatmul(state, cmd, pos, spec)) return;
    auto encoder = graphEncoder(state);
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:state.ctx->layoutMatmulPSO];
    for (int index = 0; index < 3; index++) {
        [encoder setBuffer:requireStorageBuffer(cmd[pos + 1 + index], "layout matmul") offset:0 atIndex:index];
    }
    [encoder setBuffer:layoutGpuBuffer(state) offset:0 atIndex:3];
    std::array<uint32_t, 8> params{(uint32_t)cmd[pos + 4], (uint32_t)cmd[pos + 5], spec.leftType,
        spec.rightType, spec.outputType, spec.rows, spec.columns, spec.inner};
    bindValue(encoder, params, 4);
    dispatch1D(encoder, state.ctx->layoutMatmulPSO, spec.count);
}

static NSUInteger cacheElementCount(uint32_t heads, uint32_t length, uint32_t width) {
    if ((uint64_t)heads * length > INT32_MAX / width) {
        throw std::runtime_error("Head cache shape exceeds supported storage");
    }
    return (NSUInteger)heads * length * width;
}

static id<MTLBuffer> cacheBuffer(int bufferId, NSUInteger count) {
    id<MTLBuffer> buffer = requireBuffer(bufferId, "head cache");
    if (buffer.length != count * sizeof(float)) throw std::runtime_error("Head cache buffer size mismatch");
    return buffer;
}

static MPSMatrix* headPrefixMatrix(id<MTLBuffer> buffer, int heads, int length, int width) {
    NSUInteger perRow = elementCount(heads, width) * sizeof(float);
    if (buffer.length % perRow != 0 || buffer.length / perRow < (NSUInteger)length) {
        throw std::runtime_error("Invalid head cache capacity");
    }
    // Describe the full storage; the GEMM dimensions select only the valid prefix.
    // Full descriptor rows keep multi-head GEMV strides consistent on physical Metal devices.
    return matrixBatch(buffer, (int)(buffer.length / perRow), width, heads);
}

static void encodeHeadPrefixMatmul(GraphState& state, const jint* cmd, int pos) {
    int heads = (int)positiveCount(cmd[pos + 4], "cache heads");
    int rows = (int)positiveCount(cmd[pos + 5], "query rows");
    int inner = (int)positiveCount(cmd[pos + 6], "query width");
    int length = (int)positiveCount(cmd[pos + 7], "cache length");
    int width = (int)positiveCount(cmd[pos + 8], "cache width");
    bool transposed = cmd[pos + 9] != 0;
    if (inner != (transposed ? width : length)) throw std::runtime_error("Head prefix matmul shape mismatch");
    endGraphEncoder(state);
    MPSMatrix* left = matrixBatch(cacheBuffer(cmd[pos + 1], cacheElementCount(heads, rows, inner)),
                                  rows, inner, heads);
    MPSMatrix* right = headPrefixMatrix(requireBuffer(cmd[pos + 2], "cache"), heads, length, width);
    int columns = transposed ? length : width;
    MPSMatrix* output = matrixBatch(cacheBuffer(cmd[pos + 3], cacheElementCount(heads, rows, columns)),
                                    rows, columns, heads);
    [configuredMatmulKernel(state.ctx, rows, columns, inner, false, transposed, heads)
        encodeToCommandBuffer:state.commandBuffer leftMatrix:left rightMatrix:right resultMatrix:output];
}

static std::array<uint32_t, 5> headCacheWriteParams(const jint* fields) {
    std::array<uint32_t, 5> params;
    for (int index = 0; index < 4; index++) {
        params[index] = (uint32_t)positiveCount(fields[index], "cache dimension");
    }
    int offset = fields[4];
    if (offset < 0 || (uint64_t)offset + params[1] > params[2]) {
        throw std::runtime_error("Head cache write exceeds capacity");
    }
    params[4] = (uint32_t)offset;
    return params;
}

static void encodeHeadCacheWrite(GraphState& state, const jint* cmd, int pos) {
    auto params = headCacheWriteParams(cmd + pos + 3);
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->headCacheWritePSO];
    [encoder setBuffer:cacheBuffer(cmd[pos + 1], cacheElementCount(params[0], params[1], params[3]))
                offset:0 atIndex:0];
    [encoder setBuffer:cacheBuffer(cmd[pos + 2], cacheElementCount(params[0], params[2], params[3]))
                offset:0 atIndex:1];
    [encoder setBytes:params.data() length:sizeof(params) atIndex:2];
    dispatch1D(encoder, state.ctx->headCacheWritePSO, (NSUInteger)params[0] * params[1] * params[3]);
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
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->softmaxBackwardPSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "softmax backward") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "softmax backward") offset:0 atIndex:1];
    [encoder setBuffer:requireBuffer(cmd[pos + 3], "softmax backward") offset:0 atIndex:2];
    bindValue(encoder, cols, 3);
    float scale = cmd[pos] == OP_SCALED_SOFTMAX_BACKWARD ? floatFromBits(cmd[pos + 6]) : 1.0f;
    bindValue(encoder, scale, 4);
    NSUInteger threads = rowReductionWidth(state.ctx->softmaxBackwardPSO);
    [encoder dispatchThreads:MTLSizeMake(threads, rows, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

struct TiledAttentionParams {
    uint32_t length;
    uint32_t groupSize;
    float scale;
};

struct TiledAttentionShape {
    uint32_t rows;
    uint32_t heads;
    uint32_t keyValueHeads;
    TiledAttentionParams params;
};

static TiledAttentionShape tiledAttentionShape(GraphState& state, const jint* fields) {
    uint32_t rows = (uint32_t)positiveCount(fields[0], "attention rows");
    uint32_t heads = (uint32_t)positiveCount(fields[1], "attention heads");
    uint32_t keyValueHeads = (uint32_t)positiveCount(fields[2], "attention key/value heads");
    uint32_t length = rows / heads;
    float scale = floatFromBits(fields[3]);
    if (rows % heads != 0 || heads % keyValueHeads != 0 || length < 32 || length % 32 != 0 ||
        rows > INT32_MAX / 64 || !std::isfinite(scale)) {
        throw std::runtime_error("Invalid tiled attention shape or scale");
    }
    if (state.ctx->attentionPSO == nil) throw std::runtime_error("Tiled attention is unavailable on this GPU");
    return {rows, heads, keyValueHeads, {length, heads / keyValueHeads, scale}};
}

static id<MTLBuffer> attentionBuffer(int bufferId, NSUInteger count) {
    id<MTLBuffer> buffer = requireBuffer(bufferId, "tiled attention");
    if (buffer.length != count * sizeof(float)) {
        throw std::runtime_error("Tiled attention buffer size mismatch");
    }
    return buffer;
}

static void bindAttentionValues(id<MTLComputeCommandEncoder> encoder, const jint* ids,
                                 TiledAttentionShape shape) {
    NSUInteger keyValueRows = (NSUInteger)shape.keyValueHeads * shape.params.length;
    [encoder setBuffer:attentionBuffer(ids[0], (NSUInteger)shape.rows * 64) offset:0 atIndex:0];
    [encoder setBuffer:attentionBuffer(ids[1], keyValueRows * 64) offset:0 atIndex:1];
    [encoder setBuffer:attentionBuffer(ids[2], keyValueRows * 64) offset:0 atIndex:2];
}

static void dispatchAttention(id<MTLComputeCommandEncoder> encoder, TiledAttentionShape shape,
                               NSUInteger tile, NSUInteger threads, NSUInteger heads) {
    [encoder dispatchThreadgroups:MTLSizeMake(shape.params.length / tile, heads, 1)
           threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static void encodeTiledAttention(GraphState& state, const jint* cmd, int pos) {
    TiledAttentionShape shape = tiledAttentionShape(state, cmd + pos + 6);
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->attentionPSO];
    bindAttentionValues(encoder, cmd + pos + 1, shape);
    [encoder setBuffer:attentionBuffer(cmd[pos + 4], (NSUInteger)shape.rows * 64) offset:0 atIndex:3];
    bindValue(encoder, shape.params, 4);
    [encoder setBuffer:attentionBuffer(cmd[pos + 5], (NSUInteger)shape.rows * 2) offset:0 atIndex:5];
    dispatchAttention(encoder, shape, 32, 128, shape.heads);
}

static void encodeAttentionDelta(GraphState& state, const jint* cmd, int pos, TiledAttentionShape shape) {
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->attentionDeltaPSO];
    [encoder setBuffer:attentionBuffer(cmd[pos + 4], (NSUInteger)shape.rows * 64) offset:0 atIndex:0];
    [encoder setBuffer:attentionBuffer(cmd[pos + 5], (NSUInteger)shape.rows * 64) offset:0 atIndex:1];
    [encoder setBuffer:attentionBuffer(cmd[pos + 7], shape.rows) offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(1, shape.rows, 1)
           threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

static void bindAttentionBackward(id<MTLComputeCommandEncoder> encoder, const jint* ids,
                                   TiledAttentionShape shape) {
    bindAttentionValues(encoder, ids + 1, shape);
    [encoder setBuffer:attentionBuffer(ids[5], (NSUInteger)shape.rows * 64) offset:0 atIndex:3];
    [encoder setBuffer:attentionBuffer(ids[6], (NSUInteger)shape.rows * 2) offset:0 atIndex:4];
    [encoder setBuffer:attentionBuffer(ids[7], shape.rows) offset:0 atIndex:5];
}

static void encodeTiledAttentionBackward(GraphState& state, const jint* cmd, int pos) {
    TiledAttentionShape shape = tiledAttentionShape(state, cmd + pos + 11);
    encodeAttentionDelta(state, cmd, pos, shape);
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    bindAttentionBackward(encoder, cmd + pos, shape);
    [encoder setComputePipelineState:state.ctx->attentionQueryBackwardPSO];
    [encoder setBuffer:attentionBuffer(cmd[pos + 8], (NSUInteger)shape.rows * 64) offset:0 atIndex:6];
    bindValue(encoder, shape.params, 7);
    dispatchAttention(encoder, shape, 16, 64, shape.heads);
    [encoder setComputePipelineState:state.ctx->attentionKeyValueBackwardPSO];
    NSUInteger keyValueRows = (NSUInteger)shape.keyValueHeads * shape.params.length;
    [encoder setBuffer:attentionBuffer(cmd[pos + 9], keyValueRows * 64) offset:0 atIndex:6];
    [encoder setBuffer:attentionBuffer(cmd[pos + 10], keyValueRows * 64) offset:0 atIndex:7];
    bindValue(encoder, shape.params, 8);
    dispatchAttention(encoder, shape, 16, 64, shape.keyValueHeads);
}

static void encodeLayerNormBackward(GraphState& state, const jint* cmd, int pos) {
    NSUInteger rows = positiveCount(cmd[pos + 5], "rows");
    uint32_t cols = (uint32_t)positiveCount(cmd[pos + 6], "cols");
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->layerNormBackwardPSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "layer norm backward") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "layer norm backward") offset:0 atIndex:1];
    [encoder setBuffer:requireBuffer(cmd[pos + 3], "layer norm backward") offset:0 atIndex:2];
    [encoder setBuffer:requireBuffer(cmd[pos + 4], "layer norm backward") offset:0 atIndex:3];
    bindValue(encoder, cols, 4);
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
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->rmsNormPSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "rms norm") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "rms norm") offset:0 atIndex:1];
    [encoder setBuffer:requireBuffer(cmd[pos + 3], "rms norm") offset:0 atIndex:2];
    [encoder setBuffer:requireBuffer(cmd[pos + 4], "rms norm") offset:0 atIndex:3];
    [encoder setBuffer:requireBuffer(cmd[pos + 5], "rms norm") offset:0 atIndex:4];
    bindValue(encoder, params, 5);
    NSUInteger threads = rowReductionWidth(state.ctx->rmsNormPSO);
    [encoder dispatchThreads:MTLSizeMake(threads, params.rows, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static void encodeRmsNormBackward(GraphState& state, const jint* cmd, int pos) {
    NSUInteger rows = positiveCount(cmd[pos + 6], "rms norm rows");
    uint32_t dims[] = {(uint32_t)rows, (uint32_t)positiveCount(cmd[pos + 7], "rms norm columns")};
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->rmsNormBackwardPSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "rms norm backward") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "rms norm backward") offset:0 atIndex:1];
    [encoder setBuffer:requireBuffer(cmd[pos + 3], "rms norm backward") offset:0 atIndex:2];
    [encoder setBuffer:requireBuffer(cmd[pos + 4], "rms norm backward") offset:0 atIndex:3];
    [encoder setBuffer:requireBuffer(cmd[pos + 5], "rms norm backward") offset:0 atIndex:4];
    bindValue(encoder, dims, 5);
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
    id<MTLComputeCommandEncoder> encoder = graphEncoder(state);
    [encoder setComputePipelineState:state.ctx->adamWUpdatePSO];
    [encoder setBuffer:requireBuffer(cmd[pos + 1], "AdamW weights") offset:0 atIndex:0];
    [encoder setBuffer:requireBuffer(cmd[pos + 2], "AdamW gradient") offset:0 atIndex:1];
    [encoder setBuffer:requireBuffer(cmd[pos + 3], "AdamW first moment") offset:0 atIndex:2];
    [encoder setBuffer:requireBuffer(cmd[pos + 4], "AdamW second moment") offset:0 atIndex:3];
    bindValue(encoder, params, 4);
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

static int attentionCommandWidth(int op) {
    if (op == OP_TILED_CAUSAL_ATTENTION) return 10;
    if (op == OP_TILED_CAUSAL_ATTENTION_BACKWARD) return 15;
    if (op == OP_HEAD_CACHE_WRITE) return 8;
    if (op == OP_HEAD_PREFIX_MATMUL) return 10;
    throw std::runtime_error("Unknown attention op code: " + std::to_string(op));
}

static int layoutCommandWidth(int op) {
    if (op == OP_STORAGE_COPY) return 6;
    if (op == OP_LAYOUT_TRANSFER) return 4;
    if (op == OP_LAYOUT_BINARY || op == OP_LAYOUT_UNARY || op == OP_LAYOUT_REDUCE) return 7;
    if (op == OP_LAYOUT_ASSIGN) return 5;
    if (op == OP_LAYOUT_SOFTMAX) return 6;
    if (op == OP_LAYOUT_MATMUL) return 6;
    if (op == OP_LAYOUT_INDEX_SELECT) return 6;
    if (op == OP_LAYOUT_INDEX_ADD) return 7;
    if (op == OP_LAYOUT_MAXIMUM_BACKWARD) return 10;
    throw std::runtime_error("Unknown layout op code: " + std::to_string(op));
}

static int commandWidth(int op) {
    if (op >= OP_STORAGE_COPY) return layoutCommandWidth(op);
    if (op >= OP_TILED_CAUSAL_ATTENTION) return attentionCommandWidth(op);
    if (isUnaryOp(op) || op == OP_SUM_SQUARES_SCALAR) return 4;
    if (isFiveWidthOp(op)) return 5;
    if (isBroadcastOp(op) || op == OP_CLAMP ||
        op == OP_SOFTMAX_BACKWARD || op == OP_CROSS_ENTROPY_LOSS ||
        op == OP_CROSS_ENTROPY_GRADIENT) return 6;
    if (op == OP_MATMUL || op == OP_SCATTER_ADD_ROWS ||
        op == OP_SCATTER_ADD_ROWS_ATOMIC || op == OP_LAYERNORM_BACKWARD ||
        op == OP_SPLIT_HEADS || op == OP_MERGE_HEADS || op == OP_GATHER_ROWS ||
        op == OP_CAUSAL_SOFTMAX || op == OP_CROSS_ENTROPY_FUSED ||
        op == OP_SWIGLU_BACKWARD || op == OP_SCALED_SOFTMAX_BACKWARD) return 7;
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
    else if (op == OP_SCALED_SOFTMAX_BACKWARD) encodeSoftmaxBackward(state, cmd, pos);
    else if (op == OP_TILED_CAUSAL_ATTENTION) encodeTiledAttention(state, cmd, pos);
    else if (op == OP_TILED_CAUSAL_ATTENTION_BACKWARD) encodeTiledAttentionBackward(state, cmd, pos);
    else if (op == OP_ADAMW_UPDATE) encodeAdamW(state, cmd, pos);
    else if (op == OP_HEAD_CACHE_WRITE) encodeHeadCacheWrite(state, cmd, pos);
    else if (op == OP_HEAD_PREFIX_MATMUL) encodeHeadPrefixMatmul(state, cmd, pos);
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

static bool encodeLayoutCommand(GraphState& state, const jint* cmd, int pos) {
    int op = cmd[pos];
    if (op == OP_STORAGE_COPY) encodeStorageCopy(state, cmd, pos);
    else if (op == OP_LAYOUT_TRANSFER) encodeLayoutTransfer(state, cmd, pos);
    else if (op == OP_LAYOUT_BINARY) encodeLayoutBinary(state, cmd, pos);
    else if (op == OP_LAYOUT_UNARY) encodeLayoutUnary(state, cmd, pos);
    else if (op == OP_LAYOUT_ASSIGN) encodeLayoutAssign(state, cmd, pos);
    else if (op == OP_LAYOUT_REDUCE) encodeLayoutReduce(state, cmd, pos);
    else if (op == OP_LAYOUT_SOFTMAX) encodeLayoutSoftmax(state, cmd, pos);
    else if (op == OP_LAYOUT_MAXIMUM_BACKWARD) encodeLayoutMaximumBackward(state, cmd, pos);
    else if (op == OP_LAYOUT_MATMUL) encodeLayoutMatmul(state, cmd, pos);
    else if (op == OP_LAYOUT_INDEX_SELECT) encodeLayoutIndexSelect(state, cmd, pos);
    else if (op == OP_LAYOUT_INDEX_ADD) encodeLayoutIndexAdd(state, cmd, pos);
    else return false;
    return true;
}

static void encodeCommand(GraphState& state, const jint* cmd, int pos) {
    if (encodeLayoutCommand(state, cmd, pos)) return;
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

static void checkCommandBuffer(id<MTLCommandBuffer> commands) {
    if (commands.status != MTLCommandBufferStatusError) return;
    NSString* desc = commands.error.localizedDescription ?: @"Unknown error";
    throw std::runtime_error(std::string("Metal command buffer error: ") + [desc UTF8String]);
}

static void finishCommandBuffer(GraphState& state) {
    endGraphEncoder(state);
    [state.commandBuffer commit];
    [state.commandBuffer waitUntilCompleted];
    checkCommandBuffer(state.commandBuffer);
}

static void executeGraph(const jint* commandStream, int length, const jlong* layouts = nullptr, int layoutCount = 0) {
    @autoreleasepool {
        MetalContext* ctx = getContext();
        GraphState state{ctx, graphCommands(ctx->queue), nil, {layouts, layoutCount, nil}};
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

struct LongArrayView {
    JNIEnv* env;
    jlongArray source;
    jlong* values;

    LongArrayView(JNIEnv* environment, jlongArray array) : env(environment), source(array),
            values(env->GetLongArrayElements(source, nullptr)) {}
    ~LongArrayView() {
        if (values != nullptr) env->ReleaseLongArrayElements(source, values, JNI_ABORT);
    }
};

static void flushLayoutCommands(JNIEnv* env, jintArray commands, jint length, jlongArray layouts) {
    if (!validateCommandArray(env, commands, length)) return;
    if (layouts == nullptr) {
        throwJavaRuntimeException(env, "GPU layout words cannot be null");
        return;
    }
    IntArrayView commandWords(env, commands);
    LongArrayView layoutWords(env, layouts);
    if (env->ExceptionCheck()) return;
    int layoutCount = env->GetArrayLength(layouts);
    if (commandWords.values == nullptr || (layoutCount != 0 && layoutWords.values == nullptr)) return;
    std::lock_guard<std::mutex> lock(gBufferMutex);
    executeGraph(commandWords.values, length, layoutWords.values, layoutCount);
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_metal_MetalNative_nativeFlushLayoutOps(
        JNIEnv* env, jclass, jintArray commands, jint length, jlongArray layouts) {
    @autoreleasepool {
        try { flushLayoutCommands(env, commands, length, layouts); }
        catch (const std::exception& error) { throwJavaRuntimeException(env, error.what()); }
    }
}
