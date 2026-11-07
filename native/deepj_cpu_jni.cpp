#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <array>
#include <arm_neon.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dispatch/dispatch.h>
#include <jni.h>
#include <limits>
#include <memory>
#include <vector>

namespace deepj_cpu {
template<class Body> struct RowTask {
    int count, workers;
    Body body;
    static void partition(void* context, size_t worker) {
        auto& task = *static_cast<RowTask*>(context);
        int first = int(worker * task.count / task.workers);
        int last = int((worker + 1) * task.count / task.workers);
        for (int row = first; row < last; row++) {
            task.body(row);
        }
    }
};

template<class Body> void parallelRows(int count, int workers, Body body) {
    RowTask<Body> task{count, std::max(1, std::min(count, workers)), body};
    if (task.workers == 1) RowTask<Body>::partition(&task, 0);
    else dispatch_apply_f(task.workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
            &task, RowTask<Body>::partition);
}

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

inline bool packedMatrix(int rows, int columns, int rs, int cs) {
    return (cs == 1 && (rows == 1 || rs >= columns))
        || (rs == 1 && (columns == 1 || cs >= rows));
}

inline BNNSNDArrayDescriptor halfMatrix(void* data, int rows, int columns, int rs, int cs) {
    BNNSNDArrayDescriptor result{};
    bool rowMajor = cs == 1;
    result.layout = rowMajor ? BNNSDataLayoutRowMajorMatrix : BNNSDataLayoutColumnMajorMatrix;
    result.size[0] = rowMajor ? columns : rows;
    result.size[1] = rowMajor ? rows : columns;
    result.stride[0] = 1;
    result.stride[1] = rowMajor ? rs : cs;
    result.data = data;
    result.data_type = BNNSDataTypeFloat16;
    return result;
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
inline bool halfMatrixApply(const BNNSNDArrayDescriptor& a, const BNNSNDArrayDescriptor& b,
                             const BNNSNDArrayDescriptor& output) {
    BNNSFilterParameters parameters{};
    parameters.n_threads = 1;
    ssize_t bytes = BNNSMatMulWorkspaceSize(false, false, 1, &a, &b, &output, &parameters);
    if (bytes < 0 || bytes > 4 * 1024 * 1024) return false;
    std::vector<char> workspace(bytes);
    return BNNSMatMul(false, false, 1, &a, &b, &output, workspace.data(), &parameters) == 0;
}
#pragma clang diagnostic pop

inline bool halfMatrixProduct(const void* a, const void* b, void* c, const int* s) {
    // Keep tiny grouped-attention products on their lower-overhead route.
    if (s[11] != 2 || s[12] != 2 || s[13] != 2 || s[0] < 16) return false;
    if (uint64_t(s[0]) * s[1] * s[2] < 1024 * 1024) return false;
    if (!packedMatrix(s[0], s[2], s[6], s[7]) || !packedMatrix(s[2], s[1], s[8], s[9])) return false;
    auto left = halfMatrix((void*)(static_cast<const uint16_t*>(a) + s[3]), s[0], s[2], s[6], s[7]);
    auto right = halfMatrix((void*)(static_cast<const uint16_t*>(b) + s[4]), s[2], s[1], s[8], s[9]);
    auto output = halfMatrix(static_cast<uint16_t*>(c) + s[5], s[0], s[1], s[10], 1);
    return halfMatrixApply(left, right, output);
}

inline void compactProduct(const void* a, const void* b, void* c, const int* s,
                           float* left, float* right, float* result) {
    if (halfMatrixProduct(a, b, c, s)) return;
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
    using Buffer = std::unique_ptr<float[]>;
    Buffer left, right, output;
    static Buffer allocate(size_t count) { return count ? Buffer(new float[count]) : nullptr; }
    explicit ProductScratch(const int* s)
        : left(allocate(directLeft(s) ? 0 : size_t(s[0]) * s[2])),
          right(allocate(s[0] == 1 && (s[8] == 1 || s[9] == 1) ? 0 : size_t(s[2]) * s[1])),
          output(allocate(size_t(s[0]) * s[1])) {
        // GEMM with beta zero overwrites its output; vector kernels can accumulate into it.
        if (s[0] == 1) std::fill_n(output.get(), s[1], 0.0f);
    }
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

inline void activationExponents(const float* gate, double* output, int count) {
    std::array<double, 256> input;
    for (int i = 0; i < count; i++) {
        input[i] = -double(gate[i]);
    }
    vvexp(output, input.data(), &count);
}

inline bool activationBlock(const float* gate, const float* up, float* output, int count) {
    std::array<double, 256> exponentials;
    activationExponents(gate, exponentials.data(), count);
    return activationValues(gate, up, output, exponentials.data(), count);
}

inline bool activation(const float* gate, const float* up, float* output, int count) {
    for (int first = 0; first < count; first += 256) {
        if (!activationBlock(gate + first, up + first, output + first, std::min(256, count - first))) return false;
    }
    return true;
}
inline bool compactPointwise(ProductArrays& arrays, const int* s) {
    std::array<float, 256> left, right, output;
    for (int first = 0; first < s[3]; first += 256) {
        int count = std::min(256, s[3] - first);
        pack(arrays.left.data, left.data(), 1, count, s[0] + first, 0, 1, s[4]);
        pack(arrays.right.data, right.data(), 1, count, s[1] + first, 0, 1, s[5]);
        if (s[7] != 0 && !activationBlock(left.data(), right.data(), output.data(), count)) return false;
        for (int index = 0; index < count; index++) {
            float value = s[7] == 0 ? left[index] + right[index] : output[index];
            write(arrays.output.data, s[2] + first + index, value, s[6]);
        }
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
                                      s, scratch.left.get(), scratch.right.get(), scratch.output.get());
        }
    } catch (const std::bad_alloc&) {
        env->ThrowNew(env->FindClass("java/lang/OutOfMemoryError"), "CPU tile allocation");
    }
}

static void denseProducts(deepj_cpu::ProductArrays& arrays, const std::vector<jint>& shape) {
    const jint* s = shape.data();
    auto* a = static_cast<float*>(arrays.left.data);
    auto* b = static_cast<float*>(arrays.right.data);
    auto* c = static_cast<float*>(arrays.output.data);
    for (size_t offset = 10; offset < shape.size(); offset += 3) {
        cblas_sgemm(CblasRowMajor, (s[8] & 2) ? CblasTrans : CblasNoTrans,
                (s[8] & 1) ? CblasTrans : CblasNoTrans,
                s[0], s[1], s[2], 1, a + s[offset], s[6], b + s[offset + 1], s[7],
                0, c + s[offset + 2], s[1]);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_CpuBlas_product(
        JNIEnv* env, jclass, jfloatArray left, jfloatArray right, jfloatArray output, jintArray shape) {
    try {
        std::vector<jint> s(env->GetArrayLength(shape));
        env->GetIntArrayRegion(shape, 0, (jsize)s.size(), s.data());
        if (env->ExceptionCheck()) return;
        deepj_cpu::SerialBlas serial(s[9] != 0);
        deepj_cpu::ProductArrays arrays(env, left, right, output);
        if (arrays.valid()) denseProducts(arrays, s);
    } catch (const std::bad_alloc&) {
        env->ThrowNew(env->FindClass("java/lang/OutOfMemoryError"), "CPU batch metadata allocation");
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

extern "C" JNIEXPORT jboolean JNICALL
Java_io_github_kirstenali_deepj_tensor_CpuBlas_compactPointwise(
        JNIEnv* env, jclass, jarray left, jarray right, jarray output, jintArray shape) {
    jint s[8];
    env->GetIntArrayRegion(shape, 0, 8, s);
    if (env->ExceptionCheck()) return false;
    deepj_cpu::ProductArrays arrays(env, left, right, output);
    return arrays.valid() && deepj_cpu::compactPointwise(arrays, s);
}

namespace deepj_cpu {
struct ActivationGradients {
    PinnedArray gradient, gate, up, gateGradient, upGradient;
    ActivationGradients(JNIEnv* env, jarray dy, jarray a, jarray b, jfloatArray da, jfloatArray db)
        : gradient(env, dy, JNI_ABORT), gate(env, a, JNI_ABORT, gradient.data != nullptr),
          up(env, b, JNI_ABORT, gate.data != nullptr), gateGradient(env, da, 0, up.data != nullptr),
          upGradient(env, db, 0, gateGradient.data != nullptr) {}
};

struct ActivationTile {
    std::array<float, 256> gradient, gate, up;
    std::array<double, 256> exponentials;
};

inline bool activationAdjoints(ActivationTile& tile, float* gateGradient, float* upGradient, int count) {
    #pragma clang fp contract(off)
    for (int index = 0; index < count; index++) {
        float exponential;
        if (!roundedExponential(tile.exponentials[index], exponential)) return false;
        float sigmoid = 1.0f / (1.0f + exponential), activated = tile.gate[index] * sigmoid;
        float derivative = sigmoid + activated * (1.0f - sigmoid);
        gateGradient[index] = (tile.gradient[index] * tile.up[index]) * derivative;
        upGradient[index] = tile.gradient[index] * activated;
    }
    return true;
}

inline bool compactActivationBackward(ActivationGradients& arrays, const int* s) {
    ActivationTile tile;
    for (int first = 0; first < s[4]; first += 256) {
        int count = std::min(256, s[4] - first);
        pack(arrays.gradient.data, tile.gradient.data(), 1, count, s[0] + first, 0, 1, s[5]);
        pack(arrays.gate.data, tile.gate.data(), 1, count, s[1] + first, 0, 1, s[6]);
        pack(arrays.up.data, tile.up.data(), 1, count, s[2] + first, 0, 1, s[7]);
        activationExponents(tile.gate.data(), tile.exponentials.data(), count);
        auto* da = static_cast<float*>(arrays.gateGradient.data) + s[3] + first;
        auto* db = static_cast<float*>(arrays.upGradient.data) + s[3] + first;
        if (!activationAdjoints(tile, da, db, count)) return false;
    }
    return true;
}
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_github_kirstenali_deepj_tensor_CpuBlas_compactSwiGluBackward(
        JNIEnv* env, jclass, jarray gradient, jarray gate, jarray up,
        jfloatArray gateGradient, jfloatArray upGradient, jintArray shape) {
    jint s[8];
    env->GetIntArrayRegion(shape, 0, 8, s);
    if (env->ExceptionCheck()) return false;
    deepj_cpu::ActivationGradients arrays(env, gradient, gate, up, gateGradient, upGradient);
    return arrays.upGradient.data && deepj_cpu::compactActivationBackward(arrays, s);
}

namespace deepj_normalization {
struct Arrays {
    deepj_cpu::PinnedArray input, gamma, beta, output, normalized, denominator;
    Arrays(JNIEnv* env, jarray x, jarray g, jarray b,
            jarray y, jfloatArray n, jfloatArray d)
        : input(env, x, JNI_ABORT), gamma(env, g, JNI_ABORT, input.data != nullptr),
          beta(env, b, JNI_ABORT, gamma.data != nullptr), output(env, y, 0, beta.data != nullptr),
          normalized(env, n, 0, output.data != nullptr), denominator(env, d, 0, normalized.data != nullptr) {}
};

static float mean(const float* input, int width, bool centered) {
    #pragma clang fp contract(off)
    if (!centered) return 0;
    float sum = 0;
    for (int column = 0; column < width; column++) { sum += input[column]; }
    return sum * (1.0f / width);
}

static float denominator(const float* input, int width, float average, float epsilon) {
    #pragma clang fp contract(off)
    float squares = 0;
    for (int column = 0; column < width; column++) {
        float difference = input[column] - average;
        squares += difference * difference;
    }
    float variance = squares * (1.0f / width) + epsilon;
    return static_cast<float>(std::sqrt(static_cast<double>(variance)));
}

struct Row {
    const float* input;
    const float* gamma;
    const float* beta;
    void* output;
    float* normalized;
    float* denominator;
    int width, type;
    bool centered;
};

static void normalize(Row row, float epsilon) {
    #pragma clang fp contract(off)
    float average = mean(row.input, row.width, row.centered);
    float scale = denominator(row.input, row.width, average, epsilon);
    *row.denominator = scale;
    for (int column = 0; column < row.width; column++) {
        row.normalized[column] = (row.input[column] - average) / scale;
        float value = row.normalized[column] * row.gamma[column];
        deepj_cpu::write(row.output, column, row.centered ? value + row.beta[column] : value, row.type);
    }
}

static void row(Arrays& arrays, const int* shape, int row, float epsilon) {
    int width = shape[4], base = row * width;
    normalize({static_cast<float*>(arrays.input.data) + shape[0] + base,
        static_cast<float*>(arrays.gamma.data) + shape[1], static_cast<float*>(arrays.beta.data) + shape[2],
        static_cast<float*>(arrays.output.data) + base, static_cast<float*>(arrays.normalized.data) + base,
        static_cast<float*>(arrays.denominator.data) + row, width, 1, shape[5] != 0}, epsilon);
}

static void compactRow(Arrays& arrays, const int* shape, int row, float epsilon) {
    int width = shape[4], base = row * width;
    std::array<float, 8192> gamma, beta;
    auto* input = static_cast<float*>(arrays.normalized.data) + base;
    deepj_cpu::pack(arrays.input.data, input, 1, width, shape[0] + base, 0, 1, shape[7]);
    deepj_cpu::pack(arrays.gamma.data, gamma.data(), 1, width, shape[1], 0, 1, shape[8]);
    if (shape[5]) deepj_cpu::pack(arrays.beta.data, beta.data(), 1, width, shape[2], 0, 1, shape[9]);
    auto* output = static_cast<char*>(arrays.output.data) + base * (shape[10] == 1 ? 4 : 2);
    normalize({input, gamma.data(), beta.data(), output, input,
        static_cast<float*>(arrays.denominator.data) + row, width, shape[10], shape[5] != 0}, epsilon);
}
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_CpuBlas_normalize(
        JNIEnv* env, jclass, jfloatArray input, jfloatArray gamma, jfloatArray beta, jfloatArray output,
        jfloatArray normalized, jfloatArray denominator, jintArray shape, jfloat epsilon) {
    jint s[7];
    env->GetIntArrayRegion(shape, 0, 7, s);
    if (env->ExceptionCheck()) return;
    deepj_normalization::Arrays arrays(env, input, gamma, beta, output, normalized, denominator);
    if (!arrays.denominator.data) return;
    deepj_cpu::parallelRows(s[3], s[6], [&](int row) {
        deepj_normalization::row(arrays, s, row, epsilon);
    });
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_CpuBlas_compactNormalize(
        JNIEnv* env, jclass, jarray input, jarray gamma, jarray beta, jarray output,
        jfloatArray normalized, jfloatArray denominator, jintArray shape, jfloat epsilon) {
    jint s[11];
    env->GetIntArrayRegion(shape, 0, 11, s);
    if (env->ExceptionCheck()) return;
    deepj_normalization::Arrays arrays(env, input, gamma, beta, output, normalized, denominator);
    if (!arrays.denominator.data) return;
    deepj_cpu::parallelRows(s[3], s[6], [&](int row) {
        deepj_normalization::compactRow(arrays, s, row, epsilon);
    });
}

namespace deepj_normalization {
struct Gradients {
    deepj_cpu::PinnedArray gradient, normalized, denominator, gamma, output;
    Gradients(JNIEnv* env, jarray g, jfloatArray n, jfloatArray d, jarray w, jfloatArray y)
        : gradient(env, g, JNI_ABORT), normalized(env, n, JNI_ABORT, gradient.data != nullptr),
          denominator(env, d, JNI_ABORT, normalized.data != nullptr),
          gamma(env, w, JNI_ABORT, denominator.data != nullptr), output(env, y, 0, gamma.data != nullptr) {}
};

static void gradientValues(const float* g, const float* n, const float* w, float* output,
                            float d, int width, bool centered) {
    #pragma clang fp contract(off)
    float sum = 0, dot = 0, inverse = 1.0f / d;
    for (int column = 0; column < width; column++) {
        float scaled = g[column] * w[column];
        sum += scaled;
        dot += scaled * n[column];
    }
    for (int column = 0; column < width; column++) {
        float scaled = g[column] * w[column];
        output[column] = centered ? inverse * (scaled - sum / width - n[column] * (dot / width))
            : (scaled - n[column] * (dot * (1.0f / width))) / d;
    }
}

static void gradientRow(Gradients& arrays, const int* shape, int row) {
    int width = shape[5], base = row * width;
    gradientValues(static_cast<float*>(arrays.gradient.data) + shape[0] + base,
        static_cast<float*>(arrays.normalized.data) + shape[1] + base,
        static_cast<float*>(arrays.gamma.data) + shape[3], static_cast<float*>(arrays.output.data) + base,
        static_cast<float*>(arrays.denominator.data)[shape[2] + row], width, shape[6] != 0);
}

static void compactGradientRow(Gradients& arrays, const int* shape, int row) {
    int width = shape[5], base = row * width;
    std::array<float, 8192> gradient, gamma;
    deepj_cpu::pack(arrays.gradient.data, gradient.data(), 1, width, shape[0] + base, 0, 1, shape[8]);
    deepj_cpu::pack(arrays.gamma.data, gamma.data(), 1, width, shape[3], 0, 1, shape[9]);
    gradientValues(gradient.data(), static_cast<float*>(arrays.normalized.data) + shape[1] + base,
        gamma.data(), static_cast<float*>(arrays.output.data) + base,
        static_cast<float*>(arrays.denominator.data)[shape[2] + row], width, shape[6] != 0);
}
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_CpuBlas_normalizationBackward(
        JNIEnv* env, jclass, jfloatArray gradient, jfloatArray normalized, jfloatArray denominator,
        jfloatArray gamma, jfloatArray output, jintArray shape) {
    jint s[8];
    env->GetIntArrayRegion(shape, 0, 8, s);
    if (env->ExceptionCheck()) return;
    deepj_normalization::Gradients arrays(env, gradient, normalized, denominator, gamma, output);
    if (!arrays.output.data) return;
    deepj_cpu::parallelRows(s[4], s[7], [&](int row) {
        deepj_normalization::gradientRow(arrays, s, row);
    });
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_kirstenali_deepj_tensor_CpuBlas_compactNormalizationBackward(
        JNIEnv* env, jclass, jarray gradient, jfloatArray normalized, jfloatArray denominator,
        jarray gamma, jfloatArray output, jintArray shape) {
    jint s[10];
    env->GetIntArrayRegion(shape, 0, 10, s);
    if (env->ExceptionCheck()) return;
    deepj_normalization::Gradients arrays(env, gradient, normalized, denominator, gamma, output);
    if (!arrays.output.data) return;
    deepj_cpu::parallelRows(s[4], s[7], [&](int row) {
        deepj_normalization::compactGradientRow(arrays, s, row);
    });
}

namespace deepj_softmax {
static bool exponentials(float* values, int width, float maximum, float& total) {
    std::array<double, 256> input, output;
    for (int start = 0; start < width; start += 256) {
        int count = std::min(256, width - start);
        for (int index = 0; index < count; index++) { input[index] = values[start + index] - maximum; }
        vvexp(output.data(), input.data(), &count);
        for (int index = 0; index < count; index++) {
            if (!deepj_cpu::roundedExponential(output[index], values[start + index])) return false;
            total += values[start + index];
        }
    }
    return true;
}

static bool row(const float* input, float* output, int row, int width, float scale) {
    float maximum = -INFINITY, total = 0;
    for (int column = 0; column < width; column++) {
        output[column] = column <= row % width ? input[column] * scale : -1e9f;
        if (output[column] > maximum) maximum = output[column];
    }
    if (!exponentials(output, width, maximum, total)) return false;
    for (int column = 0; column < width; column++) { output[column] /= total; }
    return true;
}

}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_github_kirstenali_deepj_tensor_CpuBlas_causalSoftmax(
        JNIEnv* env, jclass, jfloatArray input, jfloatArray output, jint rows, jint width,
        jfloat scale, jint workers, jintArray fallback) {
    deepj_cpu::PinnedArray source(env, input, JNI_ABORT);
    deepj_cpu::PinnedArray result(env, output, 0, source.data != nullptr);
    deepj_cpu::PinnedArray failed(env, fallback, 0, result.data != nullptr);
    if (!failed.data) return false;
    deepj_cpu::parallelRows(rows, workers, [&](int row) {
        static_cast<jint*>(failed.data)[row] = !deepj_softmax::row(
                static_cast<float*>(source.data) + row * width,
                static_cast<float*>(result.data) + row * width, row, width, scale);
    });
    return true;
}

namespace deepj_softmax {
static bool compactRow(const void* input, void* output, const int* shape, int row, float scale) {
    std::array<float, 4096> values;
    int width = shape[3], count = shape[5] == 0 ? width : row % width + 1;
    deepj_cpu::pack(input, values.data(), 1, count, shape[0] + row * width, 0, 1, shape[4]);
    float maximum = -INFINITY, total = 0;
    for (int column = 0; column < count; column++) {
        values[column] *= scale;
        if (values[column] > maximum) maximum = values[column];
    }
    if (!exponentials(values.data(), count, maximum, total)) return false;
    for (int column = 0; column < width; column++) {
        float value = column < count ? values[column] / total : 0.0f;
        deepj_cpu::write(output, shape[1] + row * width + column, value, shape[4]);
    }
    return true;
}
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_github_kirstenali_deepj_tensor_CpuBlas_compactSoftmax(
        JNIEnv* env, jclass, jshortArray input, jshortArray output, jintArray shape, jfloat scale, jintArray fallback) {
    jint s[7];
    env->GetIntArrayRegion(shape, 0, 7, s);
    if (env->ExceptionCheck()) return false;
    deepj_cpu::PinnedArray source(env, input, JNI_ABORT);
    deepj_cpu::PinnedArray result(env, output, 0, source.data != nullptr);
    deepj_cpu::PinnedArray failed(env, fallback, 0, result.data != nullptr);
    if (!failed.data) return false;
    deepj_cpu::parallelRows(s[2], s[6], [&](int row) {
        static_cast<jint*>(failed.data)[row] = !deepj_softmax::compactRow(source.data, result.data, s, row, scale);
    });
    return true;
}

namespace deepj_addition {
static float32x4x4_t transpose(float32x4x4_t input) {
    auto low01 = vtrn1q_f32(input.val[0], input.val[1]), high01 = vtrn2q_f32(input.val[0], input.val[1]);
    auto low23 = vtrn1q_f32(input.val[2], input.val[3]), high23 = vtrn2q_f32(input.val[2], input.val[3]);
    return {{vcombine_f32(vget_low_f32(low01), vget_low_f32(low23)),
             vcombine_f32(vget_low_f32(high01), vget_low_f32(high23)),
             vcombine_f32(vget_high_f32(low01), vget_high_f32(low23)),
             vcombine_f32(vget_high_f32(high01), vget_high_f32(high23))}};
}

static float32x4x4_t block(const float* input, int rowStride, int columnStride) {
    int stride = columnStride == 1 ? rowStride : columnStride;
    float32x4x4_t values{{vld1q_f32(input), vld1q_f32(input + stride),
                        vld1q_f32(input + 2 * stride), vld1q_f32(input + 3 * stride)}};
    return columnStride == 1 ? values : transpose(values);
}

static void rows(const float* a, const float* b, float* output, const int* s, int row, int first, int last) {
    int column = first;
    for (; last - column >= 4; column += 4) {
        auto left = block(a + row * s[5] + column * s[6], s[5], s[6]);
        auto right = block(b + row * s[7] + column * s[8], s[7], s[8]);
        for (int r = 0; r < 4; r++) {
            vst1q_f32(output + (row + r) * s[4] + column, vaddq_f32(left.val[r], right.val[r]));
        }
    }
    for (; column < last; column++) {
        for (int r = 0; r < 4; r++) {
            output[(row + r) * s[4] + column] = a[(row + r) * s[5] + column * s[6]]
                                              + b[(row + r) * s[7] + column * s[8]];
        }
    }
}

static void tile(const float* a, const float* b, float* output, const int* s, int firstRow, int firstColumn) {
    int lastRow = firstRow + std::min(64, s[3] - firstRow);
    int lastColumn = firstColumn + std::min(64, s[4] - firstColumn);
    int row = firstRow;
    for (; lastRow - row >= 4; row += 4) {
        rows(a, b, output, s, row, firstColumn, lastColumn);
    }
    for (; row < lastRow; row++) {
        for (int column = firstColumn; column < lastColumn; column++) {
            output[row * s[4] + column] = a[row * s[5] + column * s[6]] + b[row * s[7] + column * s[8]];
        }
    }
}

static void apply(const float* a, const float* b, float* output, const int* shape) {
    for (int row = 0; row < shape[3]; row += std::min(64, shape[3] - row)) {
        for (int column = 0; column < shape[4]; column += std::min(64, shape[4] - column)) {
            tile(a, b, output, shape, row, column);
        }
    }
}

struct Arrays {
    deepj_cpu::PinnedArray output, left, right;
    const float* a;
    const float* b;
    Arrays(JNIEnv* env, jfloatArray av, jfloatArray bv, jfloatArray result, bool aliasA, bool aliasB)
        : output(env, result, 0), left(env, av, JNI_ABORT, output.data && !aliasA),
          right(env, bv, JNI_ABORT, output.data && (aliasA || left.data) && !aliasB),
          a(static_cast<const float*>(aliasA ? output.data : left.data)),
          b(static_cast<const float*>(aliasB ? output.data : right.data)) {}
    bool valid() const { return output.data && a && b; }
};
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_github_kirstenali_deepj_tensor_CpuBlas_matrixAdd(
        JNIEnv* env, jclass, jfloatArray left, jfloatArray right, jfloatArray output, jintArray shape) {
    jint s[9];
    env->GetIntArrayRegion(shape, 0, 9, s);
    if (env->ExceptionCheck()) return false;
    bool aliasA = env->IsSameObject(left, output), aliasB = env->IsSameObject(right, output);
    deepj_addition::Arrays arrays(env, left, right, output, aliasA, aliasB);
    if (!arrays.valid()) return false;
    deepj_addition::apply(arrays.a + s[0], arrays.b + s[1], static_cast<float*>(arrays.output.data) + s[2], s);
    return true;
}

namespace deepj_optimizer {
struct Adam {
    float learningRate, beta1, beta2, epsilon, decay, bc1, bc2;
};

static void vectorUpdate(float* weight, const float* gradient, float* first, float* second, const Adam& p) {
    #pragma clang fp contract(off)
    auto g = vld1q_f32(gradient), w = vld1q_f32(weight);
    auto m = vaddq_f32(vmulq_n_f32(vld1q_f32(first), p.beta1), vmulq_n_f32(g, 1.0f - p.beta1));
    auto v = vaddq_f32(vmulq_n_f32(vld1q_f32(second), p.beta2), vmulq_n_f32(vmulq_f32(g, g), 1.0f - p.beta2));
    vst1q_f32(first, m);
    vst1q_f32(second, v);
    auto mHat = vdivq_f32(m, vdupq_n_f32(p.bc1)), vHat = vdivq_f32(v, vdupq_n_f32(p.bc2));
    auto update = vdivq_f32(vmulq_n_f32(mHat, p.learningRate),
                           vaddq_f32(vsqrtq_f32(vHat), vdupq_n_f32(p.epsilon)));
    if (p.decay != 0) update = vaddq_f32(update, vmulq_n_f32(w, p.learningRate * p.decay));
    vst1q_f32(weight, vsubq_f32(w, update));
}

static void scalarUpdate(float* weight, const float* gradient, float* first, float* second, const Adam& p) {
    #pragma clang fp contract(off)
    float g = *gradient;
    float m = p.beta1 * (*first) + (1.0f - p.beta1) * g;
    float v = p.beta2 * (*second) + (1.0f - p.beta2) * (g * g);
    *first = m;
    *second = v;
    float update = (p.learningRate * (m / p.bc1))
                 / (static_cast<float>(std::sqrt(double(v / p.bc2))) + p.epsilon);
    if (p.decay != 0) update += p.learningRate * p.decay * (*weight);
    *weight -= update;
}

static void apply(float* w, const float* g, float* m, float* v, int count, const Adam& p) {
    int index = 0;
    for (; count - index >= 4; index += 4) {
        vectorUpdate(w + index, g + index, m + index, v + index, p);
    }
    for (; index < count; index++) {
        scalarUpdate(w + index, g + index, m + index, v + index, p);
    }
}

static float squareBlock(const float* values, int count) {
    #pragma clang fp contract(off)
    float32x4_t sums[4] = {};
    int index = 0;
    for (; index + 16 <= count; index += 16) {
        for (int lane = 0; lane < 4; lane++) {
            auto value = vld1q_f32(values + index + 4 * lane);
            sums[lane] = vaddq_f32(sums[lane], vmulq_f32(value, value));
        }
    }
    float total = vaddvq_f32(vaddq_f32(vaddq_f32(sums[0], sums[1]), vaddq_f32(sums[2], sums[3])));
    for (; index < count; index++) {
        total += values[index] * values[index];
    }
    return total;
}

static float sumSquares(const float* values, int count, float initial) {
    double total = initial;
    for (int first = 0; first < count; first += std::min(1024, count - first)) {
        total += squareBlock(values + first, std::min(1024, count - first));
    }
    return static_cast<float>(total);
}
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_github_kirstenali_deepj_tensor_CpuBlas_adamW(
        JNIEnv* env, jclass, jfloatArray weight, jint wo, jfloatArray gradient, jint go,
        jfloatArray first, jint mo, jfloatArray second, jint vo, jint count,
        jfloat lr, jfloat beta1, jfloat beta2, jfloat epsilon, jfloat decay, jfloat bc1, jfloat bc2) {
    deepj_cpu::PinnedArray w(env, weight, 0), g(env, gradient, JNI_ABORT, w.data != nullptr);
    deepj_cpu::PinnedArray m(env, first, 0, g.data != nullptr), v(env, second, 0, m.data != nullptr);
    if (!v.data) return false;
    deepj_optimizer::apply(static_cast<float*>(w.data) + wo, static_cast<const float*>(g.data) + go,
                          static_cast<float*>(m.data) + mo, static_cast<float*>(v.data) + vo, count,
                          {lr, beta1, beta2, epsilon, decay, bc1, bc2});
    return true;
}

extern "C" JNIEXPORT jfloat JNICALL
Java_io_github_kirstenali_deepj_tensor_CpuBlas_sumSquares(JNIEnv* env, jclass, jfloatArray values, jfloat initial) {
    int count = env->GetArrayLength(values);
    deepj_cpu::PinnedArray input(env, values, JNI_ABORT);
    if (!input.data) return initial;
    return deepj_optimizer::sumSquares(static_cast<const float*>(input.data), count, initial);
}
