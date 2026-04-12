package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.*;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

/**
 * CPU vs Metal GPU chained-pipeline performance benchmarks.
 *
 * <p>Multiple lazy GPU ops are recorded into a single command buffer, then
 * one {@code materialize()} flushes them all at once.  This matches how the
 * GPU actually runs during training and is where the real speedup shows:
 * command-buffer batching amortises per-kernel dispatch overhead.
 *
 * <p>Not a rigorous benchmark (use JMH for that). Disabled by default to keep CI stable.
 * Run manually with:
 * <pre>
 *   mvn test -Dtest=MetalBackendAllOpsPerformanceTest \
 *       -Djunit.jupiter.conditions.deactivate=org.junit.jupiter.engine.extension.DisabledCondition \
 *       -DskipTests=false -Dperf.size=512 -Dperf.iters.cpu=3 -Dperf.iters.gpu=10
 * </pre>
 */
@Disabled("Manual performance test; timings vary by machine and can be unstable in CI.")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public final class MetalBackendAllOpsPerformanceTest {

    private static CpuBackend cpu;
    private static TensorBackend gpu;
    private static TensorBackend previousBackend;

    private static int N;       // matrix dimension (N × N)
    private static int IT_CPU;
    private static int IT_GPU;

    /** Collects all results for a summary table printed at the end. */
    private static final Map<String, long[]> results = new LinkedHashMap<>();

    // ── setup / teardown ──────────────────────────────────────────

    @BeforeAll
    static void setUp() {
        Assumptions.assumeTrue(MetalNative.AVAILABLE, "Metal native library not available");
        cpu = new CpuBackend();
        gpu = new MetalBackend();

        previousBackend = Tensor.backend();
        Tensor.setBackend(gpu);

        N      = intProp("perf.size", 2024);
        IT_CPU = intProp("perf.iters.cpu", 3);
        IT_GPU = intProp("perf.iters.gpu", 10);

        System.out.println("\n╔══════════════════════════════════════════════════════════════════╗");
        System.out.println("║  DeepJ — CPU vs Metal GPU Chained-Pipeline Performance         ║");
        System.out.println("╠══════════════════════════════════════════════════════════════════╣");
        System.out.printf( "║  Matrix size : %d × %d  (%,d elements)%n", N, N, (long) N * N);
        System.out.printf( "║  CPU iters   : %d   GPU iters: %d%n", IT_CPU, IT_GPU);
        System.out.println("╚══════════════════════════════════════════════════════════════════╝\n");
    }

    @AfterAll
    static void tearDown() {
        if (previousBackend != null) Tensor.setBackend(previousBackend);
        printSummary();
    }

    // ── helpers ────────────────────────────────────────────────────

    private static Tensor rand(int rows, int cols, long seed) {
        return cpu.random(rows, cols, new Random(seed));
    }

    private static int intProp(String key, int def) {
        String v = System.getProperty(key);
        if (v == null || v.isBlank()) return def;
        try { return Integer.parseInt(v.trim()); }
        catch (NumberFormatException e) { return def; }
    }

    private static long bestOfNanos(Runnable r, int iters) {
        long best = Long.MAX_VALUE;
        for (int i = 0; i < iters; i++) {
            long t0 = System.nanoTime();
            r.run();
            best = Math.min(best, System.nanoTime() - t0);
        }
        return best;
    }

    private void bench(String label, Runnable cpuOp, Runnable gpuOp) {
        for (int i = 0; i < 3; i++) { cpuOp.run(); gpuOp.run(); }  // warm-up
        long cpuNs = bestOfNanos(cpuOp, IT_CPU);
        long gpuNs = bestOfNanos(gpuOp, IT_GPU);
        results.put(label, new long[]{cpuNs, gpuNs});

        double cpuMs = cpuNs / 1_000_000.0;
        double gpuMs = gpuNs / 1_000_000.0;
        double speedup = cpuMs / gpuMs;
        String arrow = speedup >= 1.0f ? "🟢" : "🔴";
        System.out.printf("  %-40s CPU: %9.3f ms   GPU: %9.3f ms   speedup: %7.2fx  %s%n",
                label, cpuMs, gpuMs, speedup, arrow);
    }

    private static void printSummary() {
        System.out.println("\n┌──────────────────────────────────────────┬────────────┬────────────┬──────────┐");
        System.out.println("│ Pipeline                                 │   CPU (ms) │   GPU (ms) │ Speedup  │");
        System.out.println("├──────────────────────────────────────────┼────────────┼────────────┼──────────┤");
        for (var entry : results.entrySet()) {
            double cpuMs = entry.getValue()[0] / 1_000_000.0;
            double gpuMs = entry.getValue()[1] / 1_000_000.0;
            System.out.printf("│ %-40s │ %10.3f │ %10.3f │ %7.2fx │%n",
                    entry.getKey(), cpuMs, gpuMs, cpuMs / gpuMs);
        }
        System.out.println("└──────────────────────────────────────────┴────────────┴────────────┴──────────┘\n");
    }

    // ╔═════════════════════════════════════════════════════════════════╗
    // ║  MIXED CHAINS  (matmul + element-wise, scaling with depth)     ║
    // ╚═════════════════════════════════════════════════════════════════╝

    /**
     * 5 ops (1 matmul + 4 elem-wise) → 1 materialize.
     * matmul → add → gelu → multiplyScalar → exp
     */
    @Test @Order(1)
    void chain_mixed5() {
        Tensor a = rand(N, N, 70L), b = rand(N, N, 71L);
        bench("5 mixed ops (1 matmul)",
                () -> {
                    Tensor t = cpu.matmul(a, b);
                    t = cpu.add(t, a);
                    t = cpu.gelu(t);
                    t = cpu.multiplyScalar(t, 0.5f);
                    cpu.exp(t);
                },
                () -> {
                    Tensor t = gpu.matmul(a, b);
                    t = gpu.add(t, a);
                    t = gpu.gelu(t);
                    t = gpu.multiplyScalar(t, 0.5f);
                    gpu.exp(t).materialize();
                });
    }

    /**
     * 10 ops (2 matmuls + 8 elem-wise) → 1 materialize.
     */
    @Test @Order(2)
    void chain_mixed10() {
        Tensor a = rand(N, N, 72L), b = rand(N, N, 73L);
        bench("10 mixed ops (2 matmuls)",
                () -> {
                    Tensor t = cpu.matmul(a, b);
                    t = cpu.gelu(t);
                    t = cpu.multiplyScalar(t, 0.5f);
                    t = cpu.subtract(t, b);
                    t = cpu.relu(t);
                    t = cpu.matmul(t, a);
                    t = cpu.sigmoid(t);
                    t = cpu.multiply(t, b);
                    t = cpu.tanh(t);
                    cpu.neg(t);
                },
                () -> {
                    Tensor t = gpu.matmul(a, b);
                    t = gpu.gelu(t);
                    t = gpu.multiplyScalar(t, 0.5f);
                    t = gpu.subtract(t, b);
                    t = gpu.relu(t);
                    t = gpu.matmul(t, a);
                    t = gpu.sigmoid(t);
                    t = gpu.multiply(t, b);
                    t = gpu.tanh(t);
                    gpu.neg(t).materialize();
                });
    }

    /**
     * 20 ops (4 matmuls + 16 elem-wise) → 1 materialize.
     * Dispatch overhead fully amortised; matmuls dominate CPU time.
     */
    @Test @Order(3)
    void chain_mixed20() {
        Tensor a = rand(N, N, 74L), b = rand(N, N, 75L);
        bench("20 mixed ops (4 matmuls)",
                () -> {
                    Tensor t = cpu.matmul(a, b);
                    for (int i = 0; i < 19; i++) {
                        t = switch (i % 5) {
                            case 0 -> cpu.gelu(t);
                            case 1 -> cpu.matmul(t, a);
                            case 2 -> cpu.add(t, b);
                            case 3 -> cpu.sigmoid(t);
                            default -> cpu.subtract(t, a);
                        };
                    }
                },
                () -> {
                    Tensor t = gpu.matmul(a, b);
                    for (int i = 0; i < 19; i++) {
                        t = switch (i % 5) {
                            case 0 -> gpu.gelu(t);
                            case 1 -> gpu.matmul(t, a);
                            case 2 -> gpu.add(t, b);
                            case 3 -> gpu.sigmoid(t);
                            default -> gpu.subtract(t, a);
                        };
                    }
                    t.materialize();
                });
    }

    /**
     * 50 ops (10 matmuls + 40 elem-wise) → 1 materialize.
     * Deep pipeline — shows full batching advantage at scale.
     */
    @Test @Order(4)
    void chain_mixed50() {
        Tensor a = rand(N, N, 76L), b = rand(N, N, 77L);
        bench("50 mixed ops (10 matmuls)",
                () -> {
                    Tensor t = cpu.matmul(a, b);
                    for (int i = 0; i < 49; i++) {
                        t = switch (i % 5) {
                            case 0 -> cpu.gelu(t);
                            case 1 -> cpu.matmul(t, a);
                            case 2 -> cpu.add(t, b);
                            case 3 -> cpu.sigmoid(t);
                            default -> cpu.subtract(t, a);
                        };
                    }
                },
                () -> {
                    Tensor t = gpu.matmul(a, b);
                    for (int i = 0; i < 49; i++) {
                        t = switch (i % 5) {
                            case 0 -> gpu.gelu(t);
                            case 1 -> gpu.matmul(t, a);
                            case 2 -> gpu.add(t, b);
                            case 3 -> gpu.sigmoid(t);
                            default -> gpu.subtract(t, a);
                        };
                    }
                    t.materialize();
                });
    }

    // ╔═════════════════════════════════════════════════════════════════╗
    // ║  MODEL-LAYER PIPELINES                                         ║
    // ╚═════════════════════════════════════════════════════════════════╝

    /**
     * Linear-layer forward: matmul → add → gelu  (3 GPU ops → 1 materialize).
     * Bias is pre-expanded to N×N so the add stays on GPU.
     */
    @Test @Order(10)
    void chain_linearForward() {
        Tensor x = rand(N, N, 76L), W = rand(N, N, 77L);
        Tensor bias = rand(N, N, 78L);
        bench("linear fwd (3 ops)",
                () -> {
                    Tensor h = cpu.matmul(x, W);
                    h = cpu.add(h, bias);
                    cpu.gelu(h);
                },
                () -> {
                    Tensor h = gpu.matmul(x, W);
                    h = gpu.add(h, bias);
                    gpu.gelu(h).materialize();
                });
    }

    /**
     * Self-attention scores: Q·K → scale → softmax → ·V  (4 GPU ops → 1 materialize).
     */
    @Test @Order(11)
    void chain_attentionForward() {
        Tensor Q = rand(N, N, 79L), K = rand(N, N, 80L), V = rand(N, N, 81L);
        float scale = (float) (1.0 / Math.sqrt(N));
        bench("attention fwd (4 ops)",
                () -> {
                    Tensor scores = cpu.matmul(Q, K);
                    scores = cpu.multiplyScalar(scores, scale);
                    Tensor probs  = cpu.softmaxRows(scores);
                    cpu.matmul(probs, V);
                },
                () -> {
                    Tensor scores = gpu.matmul(Q, K);
                    scores = gpu.multiplyScalar(scores, scale);
                    Tensor probs  = gpu.softmaxRows(scores);
                    gpu.matmul(probs, V).materialize();
                });
    }

    /**
     * Forward + loss gradient: matmul → gelu → matmul → crossEntropyGradient.
     * crossEntropyGradient internally chains softmaxRows + subtract + multiplyScalar,
     * so the GPU records ~6 kernels total before the single materialize.
     */
    @Test @Order(12)
    void chain_forwardAndLossGrad() {
        Tensor x = rand(N, N, 82L), W1 = rand(N, N, 83L), W2 = rand(N, N, 84L);
        int[] targets = randomTargets(N, N, 85L);
        bench("fwd + loss grad (6 ops)",
                () -> {
                    Tensor h = cpu.matmul(x, W1);
                    h = cpu.gelu(h);
                    Tensor logits = cpu.matmul(h, W2);
                    cpu.crossEntropyGradient(logits, targets);
                },
                () -> {
                    Tensor h = gpu.matmul(x, W1);
                    h = gpu.gelu(h);
                    Tensor logits = gpu.matmul(h, W2);
                    gpu.crossEntropyGradient(logits, targets).materialize();
                });
    }

    /**
     * Two-layer backward pass through linear layers:
     *   softmaxBackward → matmul(dLogits, W2) → geluBackward →
     *   matmul(dH, W1) → subtract → multiplyScalar
     * (6 GPU ops including 2 matmuls → 1 materialize).
     */
    @Test @Order(13)
    void chain_backward() {
        Tensor grad   = rand(N, N, 86L);
        Tensor smOut  = cpu.softmaxRows(rand(N, N, 87L));
        Tensor W1     = rand(N, N, 88L);
        Tensor W2     = rand(N, N, 89L);
        Tensor hPreAct = rand(N, N, 90L);
        Tensor offset = rand(N, N, 91L);
        bench("backward (6 ops, 2 matmuls)",
                () -> {
                    Tensor d = cpu.softmaxBackward(grad, smOut);
                    d = cpu.matmul(d, W2);
                    d = cpu.geluBackward(hPreAct, d);
                    d = cpu.matmul(d, W1);
                    d = cpu.subtract(d, offset);
                    cpu.multiplyScalar(d, 0.5f);
                },
                () -> {
                    Tensor d = gpu.softmaxBackward(grad, smOut);
                    d = gpu.matmul(d, W2);
                    d = gpu.geluBackward(hPreAct, d);
                    d = gpu.matmul(d, W1);
                    d = gpu.subtract(d, offset);
                    gpu.multiplyScalar(d, 0.5f).materialize();
                });
    }

    // ╔═════════════════════════════════════════════════════════════════╗
    // ║  FULL TRAINING-STEP PIPELINES                                  ║
    // ╚═════════════════════════════════════════════════════════════════╝

    /**
     * Mini training step: forward + softmax + scale + 2× adamW.
     * ~9 GPU ops recorded, 1 materialize at the end.
     */
    @Test @Order(20)
    void chain_miniTrainStep() {
        Tensor x  = rand(N, N, 91L);
        Tensor W1 = rand(N, N, 92L), W2 = rand(N, N, 93L);
        Tensor g1 = rand(N, N, 94L), g2 = rand(N, N, 95L);

        Tensor m1 = cpu.zeros(N, N), v1 = cpu.zeros(N, N);
        Tensor m2 = cpu.zeros(N, N), v2 = cpu.zeros(N, N);

        Tensor W1G = clone(W1), W2G = clone(W2);
        Tensor g1G = clone(g1), g2G = clone(g2);
        Tensor m1G = cpu.zeros(N, N), v1G = cpu.zeros(N, N);
        Tensor m2G = cpu.zeros(N, N), v2G = cpu.zeros(N, N);

        bench("mini train step (9 ops)",
                () -> {
                    Tensor h = cpu.matmul(x, W1);
                    h = cpu.gelu(h);
                    Tensor logits = cpu.matmul(h, W2);
                    cpu.softmaxRows(logits);
                    cpu.multiplyScalar(logits, 0.1f);
                    cpu.adamWUpdate(W1, g1, m1, v1, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, 0.9f, 0.999f);
                    cpu.adamWUpdate(W2, g2, m2, v2, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, 0.9f, 0.999f);
                },
                () -> {
                    Tensor h = gpu.matmul(x, W1G);
                    h = gpu.gelu(h);
                    Tensor logits = gpu.matmul(h, W2G);
                    gpu.softmaxRows(logits);
                    gpu.multiplyScalar(logits, 0.1f);
                    gpu.adamWUpdate(W1G, g1G, m1G, v1G, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, 0.9f, 0.999f);
                    gpu.adamWUpdate(W2G, g2G, m2G, v2G, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, 0.9f, 0.999f);
                    W1G.materialize();
                });
    }

    /**
     * Full forward + backward + optimise (two linear layers).
     *   Forward:  matmul → gelu → matmul → crossEntropyGradient (3 internal)
     *   Backward: matmul → geluBackward → matmul
     *   Optimise: 2× adamWUpdate
     * Total: ~13 GPU ops recorded, 1 materialize.
     */
    @Test @Order(21)
    void chain_fullTrainStep() {
        Tensor x  = rand(N, N, 96L);
        Tensor W1 = rand(N, N, 97L), W2 = rand(N, N, 98L);
        int[] targets = randomTargets(N, N, 99L);
        Tensor hPreAct = rand(N, N, 100L);

        Tensor m1 = cpu.zeros(N, N), v1 = cpu.zeros(N, N);
        Tensor m2 = cpu.zeros(N, N), v2 = cpu.zeros(N, N);

        Tensor W1G = clone(W1), W2G = clone(W2);
        Tensor hPreActG = clone(hPreAct);
        Tensor m1G = cpu.zeros(N, N), v1G = cpu.zeros(N, N);
        Tensor m2G = cpu.zeros(N, N), v2G = cpu.zeros(N, N);

        bench("full train step (13 ops)",
                () -> {
                    // forward
                    Tensor h = cpu.matmul(x, W1);
                    h = cpu.gelu(h);
                    Tensor logits = cpu.matmul(h, W2);
                    // loss gradient (softmax + subtract + scale)
                    Tensor dLogits = cpu.crossEntropyGradient(logits, targets);
                    // backward
                    Tensor dH = cpu.matmul(dLogits, W2);
                    dH = cpu.geluBackward(hPreAct, dH);
                    Tensor dX = cpu.matmul(dH, W1);
                    // optimise
                    cpu.adamWUpdate(W1, dX, m1, v1, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, 0.9f, 0.999f);
                    cpu.adamWUpdate(W2, dLogits, m2, v2, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, 0.9f, 0.999f);
                },
                () -> {
                    // forward
                    Tensor h = gpu.matmul(x, W1G);
                    h = gpu.gelu(h);
                    Tensor logits = gpu.matmul(h, W2G);
                    // loss gradient (softmax + subtract + scale — all lazy)
                    Tensor dLogits = gpu.crossEntropyGradient(logits, targets);
                    // backward
                    Tensor dH = gpu.matmul(dLogits, W2G);
                    dH = gpu.geluBackward(hPreActG, dH);
                    Tensor dX = gpu.matmul(dH, W1G);
                    // optimise — still lazy
                    gpu.adamWUpdate(W1G, dX, m1G, v1G, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, 0.9f, 0.999f);
                    gpu.adamWUpdate(W2G, dLogits, m2G, v2G, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, 0.9f, 0.999f);
                    // single flush
                    W1G.materialize();
                });
    }

    /**
     * In-place grad-style loop:
     *   repeat { addInPlace + multiplyScalarInPlace }
     *
     * This specifically guards against regressions where in-place ops on Metal
     * accidentally force CPU materialization/copies.
     */
    @Test @Order(22)
    void chain_inPlaceGradAccumulation() {
        Tensor delta = rand(N, N, 101L);
        int steps = intProp("perf.inplace.steps", 100);

        bench("in-place grad accum (2*steps ops)",
                () -> {
                    Tensor g = cpu.zeros(N, N);
                    for (int i = 0; i < steps; i++) {
                        cpu.addInPlace(g, delta);
                        cpu.multiplyScalarInPlace(g, 0.99f);
                    }
                },
                () -> {
                    Tensor g = cpu.zeros(N, N);
                    for (int i = 0; i < steps; i++) {
                        gpu.addInPlace(g, delta);
                        gpu.multiplyScalarInPlace(g, 0.99f);
                    }
                    g.materialize();
                });
    }

    // ═══════════════════════════════════════════════════════════════
    //  UTILITIES
    // ═══════════════════════════════════════════════════════════════

    private static int[] randomTargets(int count, int range, long seed) {
        Random rng = new Random(seed);
        int[] targets = new int[count];
        for (int i = 0; i < count; i++) targets[i] = rng.nextInt(range);
        return targets;
    }

    private static Tensor clone(Tensor t) {
        return new Tensor(t);
    }
}

