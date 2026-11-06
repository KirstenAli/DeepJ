#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <array>
#include <arm_neon.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <jni.h>
#include <limits>
#include <vector>

namespace deepj_cpu {
template<int Type> inline float read(const void* data, int index) {
    if constexpr (Type == 1) return static_cast<const float*>(data)[index];
    if constexpr (Type == 2) return static_cast<const _Float16*>(data)[index];
    uint32_t bits = uint32_t(static_cast<const uint16_t*>(data)[index]) << 16;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

template<int Type> inline float32x4_t load(const void* data, int index) {
    if constexpr (Type == 1) return vld1q_f32(static_cast<const float*>(data) + index);
    if constexpr (Type == 2) return vcvt_f32_f16(vld1_f16(static_cast<const float16_t*>(data) + index));
    return vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vld1_u16(static_cast<const uint16_t*>(data) + index)), 16));
}

inline void write(void* data, int index, float value, int type) {
    if (type == 1) {
        static_cast<float*>(data)[index] = value;
        return;
    }
    if (type == 2) {
        static_cast<_Float16*>(data)[index] = static_cast<_Float16>(value);
        return;
    }
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    uint16_t result = std::isnan(value) ? (bits >> 16) | 0x0040 : (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
    static_cast<uint16_t*>(data)[index] = result;
}
template<int Type> inline float dot(const float* a, const void* b, int base, int size) {
    float32x4_t sums[8] = {};
    int k = 0;
    for (; k + 32 <= size; k += 32) {
        #pragma unroll
        for (int lane = 0; lane < 8; lane++) {
            sums[lane] = vfmaq_f32(sums[lane], vld1q_f32(a + k + lane * 4), load<Type>(b, base + k + lane * 4));
        }
    }
    for (int width = 4; width > 0; width /= 2) {
        for (int lane = 0; lane < width; lane++) {
            sums[lane] = vaddq_f32(sums[lane], sums[lane + width]);
        }
    }
    float value = vaddvq_f32(sums[0]);
    for (; k < size; k++) {
        value += a[k] * read<Type>(b, base + k);
    }
    return value;
}

template<int Type> inline void columns(const float* a, const void* b, float* c, const int* s) {
    for (int column = 0; column < s[1]; column++) {
        c[column] = dot<Type>(a, b, s[4] + column * s[9], s[2]);
    }
}

template<int Type> inline void rowValues(const float* a, const void* b, float* c, const int* s, int k) {
    int column = 0, base = s[4] + k * s[8];
    float32x4_t value = vdupq_n_f32(a[k]);
    for (; column + 4 <= s[1]; column += 4) {
        vst1q_f32(c + column, vfmaq_f32(vld1q_f32(c + column), value, load<Type>(b, base + column)));
    }
    for (; column < s[1]; column++) {
        c[column] += a[k] * read<Type>(b, base + column);
    }
}

template<int Type> inline void block(const float* a, const void* b, float* c, const int* s,
                                     int column, int start, int end) {
    float32x4_t sums[16];
    for (int lane = 0; lane < 16; lane++) {
        sums[lane] = vld1q_f32(c + column + lane * 4);
    }
    for (int k = start; k < end; k++) {
        float32x4_t value = vdupq_n_f32(a[k]);
        int base = s[4] + k * s[8] + column;
        #pragma unroll
        for (int lane = 0; lane < 16; lane++) {
            sums[lane] = vfmaq_f32(sums[lane], value, load<Type>(b, base + lane * 4));
        }
    }
    for (int lane = 0; lane < 16; lane++) {
        vst1q_f32(c + column + lane * 4, sums[lane]);
    }
}

template<int Type> inline void rows(const float* a, const void* b, float* c, const int* s) {
    int tail[14], column = s[1] / 64 * 64;
    std::copy(s, s + 14, tail);
    tail[1] -= column;
    tail[4] += column;
    for (int start = 0; start < s[2]; start += 32) {
        int end = std::min(start + 32, s[2]);
        for (int col = 0; col < column; col += 64) {
            block<Type>(a, b, c, s, col, start, end);
        }
        if (tail[1]) {
            for (int k = start; k < end; k++) {
                rowValues<Type>(a, b, c + column, tail, k);
            }
        }
    }
}

template<int Type> inline bool vectorProduct(const float* a, const void* b, float* c, const int* s) {
    if (s[0] != 1) return false;
    if (s[8] == 1) {
        columns<Type>(a, b, c, s);
        return true;
    }
    if (s[9] != 1) return false;
    rows<Type>(a, b, c, s);
    return true;
}

inline bool vectorProduct(const float* a, const void* b, float* c, const int* s) {
    if (s[12] == 1) return vectorProduct<1>(a, b, c, s);
    return s[12] == 2 ? vectorProduct<2>(a, b, c, s) : vectorProduct<3>(a, b, c, s);
}

template<int Type> inline void packRow(const void* data, float* packed, int columns, int offset, int stride) {
    int column = 0;
    if (stride == 1) {
        for (; column + 4 <= columns; column += 4) {
            vst1q_f32(packed + column, load<Type>(data, offset + column));
        }
    }
    for (; column < columns; column++) {
        packed[column] = read<Type>(data, offset + column * stride);
    }
}

template<int Type> inline void pack(const void* data, float* packed, int rows, int columns,
                                    int offset, int rs, int cs) {
    for (int row = 0; row < rows; row++) {
        packRow<Type>(data, packed + row * columns, columns, offset + row * rs, cs);
    }
}

inline void pack(const void* data, float* packed, int rows, int columns, int offset, int rs, int cs, int type) {
    if (type == 1) pack<1>(data, packed, rows, columns, offset, rs, cs);
    else if (type == 2) pack<2>(data, packed, rows, columns, offset, rs, cs);
    else pack<3>(data, packed, rows, columns, offset, rs, cs);
}

inline void matrixProduct(const float* left, const void* b, float* right, float* result, const int* s) {
    bool transpose = s[8] == 1;
    if (transpose) pack(b, right, s[1], s[2], s[4], s[9], s[8], s[12]);
    else pack(b, right, s[2], s[1], s[4], s[8], s[9], s[12]);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, transpose ? CblasTrans : CblasNoTrans,
                s[0], s[1], s[2], 1, left, s[2], right, transpose ? s[2] : s[1], 0, result, s[1]);
}

inline bool directLeft(const int* s) {
    return s[11] == 1 && s[7] == 1 && (s[0] == 1 || s[6] == s[2]);
}

inline const float* leftValues(const void* a, float* packed, const int* s) {
    if (directLeft(s)) return static_cast<const float*>(a) + s[3];
    pack(a, packed, s[0], s[2], s[3], s[6], s[7], s[11]);
    return packed;
}

inline void compactProduct(const void* a, const void* b, void* c, const int* s,
                           float* left, float* right, float* result) {
    const float* input = leftValues(a, left, s);
    if (!vectorProduct(input, b, result, s)) {
        matrixProduct(input, b, right, result, s);
    }
    for (int row = 0; row < s[0]; row++) {
        for (int col = 0; col < s[1]; col++) {
            write(c, s[5] + row * s[10] + col, result[row * s[1] + col], s[13]);
        }
    }
}
struct PinnedArray {
    JNIEnv* env;
    jarray array;
    void* data;
    int mode;
    PinnedArray(JNIEnv* e, jarray a, int m, bool acquire = true)
        : env(e), array(a), data(acquire ? e->GetPrimitiveArrayCritical(a, nullptr) : nullptr), mode(m) {}
    ~PinnedArray() { if (data) env->ReleasePrimitiveArrayCritical(array, data, mode); }
    PinnedArray(const PinnedArray&) = delete;
    PinnedArray& operator=(const PinnedArray&) = delete;
};

struct ProductArrays {
    PinnedArray left, right, output;
    ProductArrays(JNIEnv* env, jarray a, jarray b, jarray c)
        : left(env, a, JNI_ABORT), right(env, b, JNI_ABORT, left.data != nullptr),
          output(env, c, 0, right.data != nullptr) {}
    bool valid() const { return output.data != nullptr; }
};

struct SerialBlas {
    BLAS_THREADING previous = BLASGetThreading();
    explicit SerialBlas(bool serial) { if (serial) BLASSetThreading(BLAS_THREADING_SINGLE_THREADED); }
    ~SerialBlas() { BLASSetThreading(previous); }
};

struct ProductScratch {
    std::vector<float> left, right, output;
    explicit ProductScratch(const int* s)
        : left(directLeft(s) ? 0 : size_t(s[0]) * s[2]),
          right(s[0] == 1 && (s[8] == 1 || s[9] == 1) ? 0 : size_t(s[2]) * s[1]),
          output(size_t(s[0]) * s[1]) {}
};

// Keep the Java fallback near FLOAT32 rounding boundaries, far beyond double-library rounding noise.
inline bool roundedExponential(double value, float& result) {
    if (!std::isfinite(value) || value >= std::numeric_limits<float>::max()) return false;
    result = static_cast<float>(value);
    uint32_t bits;
    std::memcpy(&bits, &result, sizeof(bits));
    bits += value >= result ? 1 : -1;
    float neighbor;
    std::memcpy(&neighbor, &bits, sizeof(neighbor));
    double boundary = (double(result) + neighbor) * 0.5;
    double margin = std::max(value, double(std::numeric_limits<float>::denorm_min())) * 0x1p-40;
    return std::abs(value - boundary) > margin;
}

inline bool activationValues(const float* gate, const float* up, float* output,
                              const double* exponentials, int count) {
    for (int i = 0; i < count; i++) {
        float exponential;
        if (!roundedExponential(exponentials[i], exponential)) return false;
        float sigmoid = 1.0f / (1.0f + exponential);
        output[i] = (gate[i] * sigmoid) * up[i];
    }
    return true;
}

inline bool activationBlock(const float* gate, const float* up, float* output, int count) {
    std::array<double, 256> input, exponentials;
    for (int i = 0; i < count; i++) {
        input[i] = -double(gate[i]);
    }
    vvexp(exponentials.data(), input.data(), &count);
    return activationValues(gate, up, output, exponentials.data(), count);
}

inline bool activation(const float* gate, const float* up, float* output, int count) {
    for (int first = 0; first < count; first += 256) {
        if (!activationBlock(gate + first, up + first, output + first, std::min(256, count - first))) return false;
    }
    return true;
}
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_CpuBlas_compactProduct(
        JNIEnv* env, jclass, jarray left, jarray right, jarray output, jintArray shape) {
    jint s[14];
    env->GetIntArrayRegion(shape, 0, 14, s);
    if (env->ExceptionCheck()) return;
    try {
        deepj_cpu::ProductScratch scratch(s);
        deepj_cpu::SerialBlas serial(true);
        deepj_cpu::ProductArrays arrays(env, left, right, output);
        if (arrays.valid()) {
            deepj_cpu::compactProduct(arrays.left.data, arrays.right.data, arrays.output.data,
                                      s, scratch.left.data(), scratch.right.data(), scratch.output.data());
        }
    } catch (const std::bad_alloc&) {
        env->ThrowNew(env->FindClass("java/lang/OutOfMemoryError"), "CPU tile allocation");
    }
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_CpuBlas_product(
        JNIEnv* env, jclass, jfloatArray left, jfloatArray right, jfloatArray output, jintArray shape) {
    jint s[10];
    env->GetIntArrayRegion(shape, 0, 10, s);
    if (env->ExceptionCheck()) return;
    deepj_cpu::SerialBlas serial(s[9] != 0);
    deepj_cpu::ProductArrays arrays(env, left, right, output);
    if (arrays.valid()) {
        auto* a = static_cast<float*>(arrays.left.data);
        auto* b = static_cast<float*>(arrays.right.data);
        auto* c = static_cast<float*>(arrays.output.data);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, s[8] ? CblasTrans : CblasNoTrans,
                    s[0], s[1], s[2], 1, a + s[3], s[6], b + s[4], s[7], 0, c + s[5], s[1]);
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_github_kirstenali_deepj_tensor_CpuBlas_swiGlu(JNIEnv* env, jclass, jfloatArray gate, jint go,
        jfloatArray up, jint uo, jfloatArray output, jint oo, jint count) {
    deepj_cpu::ProductArrays arrays(env, gate, up, output);
    if (!arrays.valid()) return false;
    const auto* a = static_cast<const float*>(arrays.left.data) + go;
    const auto* b = static_cast<const float*>(arrays.right.data) + uo;
    auto* c = static_cast<float*>(arrays.output.data) + oo;
    return deepj_cpu::activation(a, b, c, count);
}
