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
using uint2 = uint __attribute__((ext_vector_type(2)));
using uint3 = uint __attribute__((ext_vector_type(3)));
using uint4 = uint __attribute__((ext_vector_type(4)));
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
