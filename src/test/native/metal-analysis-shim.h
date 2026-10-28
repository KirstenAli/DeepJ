// Analysis-only C++ declarations for Metal syntax. Never compiled into DeepJ.
#include <algorithm>
#include <atomic>
#include <cmath>

#define kernel
#define device
#define constant
#define thread
#define threadgroup

namespace metal {
using uint = unsigned int;
using ulong = unsigned long;
using ushort = unsigned short;
using half = _Float16;
using bfloat = __bf16;
using uint2 = uint __attribute__((ext_vector_type(2)));
using uint3 = uint __attribute__((ext_vector_type(3)));
using uint4 = uint __attribute__((ext_vector_type(4)));
using ulong2 = unsigned long __attribute__((ext_vector_type(2)));
struct simdgroup_float8x8 {
    simdgroup_float8x8();
    explicit simdgroup_float8x8(float);
};
template<class T, int Columns, int Rows> using simdgroup_matrix = simdgroup_float8x8;
template<class T> void simdgroup_load(simdgroup_float8x8&, const T*, uint, ulong2 = ulong2(0), bool = false);
void simdgroup_store(const simdgroup_float8x8&, float*, uint);
void simdgroup_multiply_accumulate(simdgroup_float8x8&, const simdgroup_float8x8&,
                                  const simdgroup_float8x8&, const simdgroup_float8x8&);
float simd_sum(float);
uint clz(uint);
struct float2 {
    float x;
    float y;
    explicit float2(float);
    float2(float, float);
};
using atomic_uint = std::atomic_uint;
using std::abs;
using std::exp;
using std::log;
using std::pow;
using std::sqrt;
using std::tanh;
using std::atomic_load_explicit;
using std::atomic_compare_exchange_weak_explicit;
using std::memory_order_relaxed;
enum class mem_flags { mem_threadgroup };
void threadgroup_barrier(mem_flags);
float rsqrt(float);
template<class To, class From> To as_type(From);
template<class A, class B> auto max(A a, B b) -> decltype(a + b);
template<class A, class B> auto min(A a, B b) -> decltype(a + b);
template<class T> T clamp(T, T, T);
}
