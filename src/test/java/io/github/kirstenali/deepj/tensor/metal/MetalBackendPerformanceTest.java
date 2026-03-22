package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.Random;

/**
 * Manual CPU vs Metal timing sanity check (matmul only).
 *
 * <p>Not a rigorous benchmark (use JMH for that). Disabled by default to keep CI stable.
 */
@Disabled("Manual performance test; timings vary by machine and can be unstable in CI.")
public final class MetalBackendPerformanceTest {

    private static TensorBackend cpu;
    private static TensorBackend gpu;

    @BeforeAll
    static void setUp() {
        Assumptions.assumeTrue(MetalNative.AVAILABLE, "Metal native library not available");
        cpu = new CpuBackend();
        gpu = new MetalBackend();
    }

    @Test
    void cpuVsGpuTimings() {
        int mm = intProp("perf.matmul", 512);
        int itCpu = intProp("perf.iters.matmul.cpu", 3);
        int itGpu = intProp("perf.iters.matmul.gpu", 10);

        Tensor a = cpu.random(mm, mm, new Random(1L));
        Tensor b = cpu.random(mm, mm, new Random(2L));

        warmUp(a, b);

        System.out.println("\n=== DeepJ CPU vs Metal GPU timings (matmul only) ===");
        System.out.println("A=" + mm + "x" + mm + ", B=" + mm + "x" + mm);

        long cpuMatmul = bestOfNanos(() -> cpu.matmul(a, b), itCpu);
        long gpuMatmul = bestOfNanos(() -> gpu.matmul(a, b), itGpu);
        print("matmul", cpuMatmul, gpuMatmul);

        System.out.println("=====================================\n");
    }

    private static void warmUp(Tensor a, Tensor b) {
        for (int i = 0; i < 3; i++) {
            cpu.matmul(a, b);
            gpu.matmul(a, b);
        }
    }

    private static int intProp(String key, int def) {
        String v = System.getProperty(key);
        if (v == null || v.isBlank()) return def;
        try {
            return Integer.parseInt(v.trim());
        } catch (NumberFormatException ignored) {
            return def;
        }
    }

    /** Returns the best (minimum) time across {@code iters} runs to reduce noise. */
    private static long bestOfNanos(Runnable r, int iters) {
        long best = Long.MAX_VALUE;
        for (int i = 0; i < iters; i++) {
            long t0 = System.nanoTime();
            r.run();
            best = Math.min(best, System.nanoTime() - t0);
        }
        return best;
    }

    private static void print(String label, long cpuNanos, long gpuNanos) {
        double cpuMs = cpuNanos / 1_000_000.0;
        double gpuMs = gpuNanos / 1_000_000.0;
        System.out.printf(
                "%-10s  CPU: %8.3f ms   GPU: %8.3f ms   speedup: %6.2fx%n",
                label, cpuMs, gpuMs, cpuMs / gpuMs
        );
    }
}
