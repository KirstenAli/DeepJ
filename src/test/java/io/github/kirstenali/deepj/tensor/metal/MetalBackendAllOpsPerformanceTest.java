package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.*;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

@Disabled("Manual performance test; timings vary by machine and can be unstable in CI.")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public final class MetalBackendAllOpsPerformanceTest {

    private static CpuBackend cpu;
    private static TensorBackend gpu;
    private static TensorBackend previousBackend;

    private static int N;
    private static int IT_CPU;
    private static int IT_GPU;

    private static final Map<String, long[]> results = new LinkedHashMap<>();

    @BeforeAll
    static void setUp() {
        Assumptions.assumeTrue(MetalBackend.isAvailable(), "Metal device not available");
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
        for (int i = 0; i < 3; i++) { cpuOp.run(); gpuOp.run(); }
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

    @Test @Order(2)
    void chain_mixed10() {
        Tensor a = rand(N, N, 72L), b = rand(N, N, 73L);
        bench("10 mixed ops (2 matmuls)",
                () -> mixedTen(cpu, a, b),
                () -> mixedTen(gpu, a, b).materialize());
    }

    @Test @Order(3)
    void chain_mixed20() {
        Tensor a = rand(N, N, 74L), b = rand(N, N, 75L);
        bench("20 mixed ops (4 matmuls)",
                () -> mixedChain(cpu, a, b, 19),
                () -> mixedChain(gpu, a, b, 19).materialize());
    }

    @Test @Order(4)
    void chain_mixed50() {
        Tensor a = rand(N, N, 76L), b = rand(N, N, 77L);
        bench("50 mixed ops (10 matmuls)",
                () -> mixedChain(cpu, a, b, 49),
                () -> mixedChain(gpu, a, b, 49).materialize());
    }

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

    @Test @Order(13)
    void chain_backward() {
        Tensor grad   = rand(N, N, 86L);
        Tensor smOut  = cpu.softmaxRows(rand(N, N, 87L));
        Tensor W1     = rand(N, N, 88L);
        Tensor W2     = rand(N, N, 89L);
        Tensor hPreAct = rand(N, N, 90L);
        Tensor offset = rand(N, N, 91L);
        bench("backward (6 ops, 2 matmuls)",
                () -> backward(cpu, grad, smOut, W1, W2, hPreAct, offset),
                () -> backward(gpu, grad, smOut, W1, W2, hPreAct, offset).materialize());
    }

    @Test @Order(20)
    void chain_miniTrainStep() {
        Tensor x = rand(N, N, 91L);
        MiniState cpuState = miniState();
        MiniState gpuState = copy(cpuState);
        bench("mini train step (9 ops)",
                () -> miniTrainStep(cpu, x, cpuState),
                () -> miniTrainStep(gpu, x, gpuState).materialize());
    }

    @Test @Order(21)
    void chain_fullTrainStep() {
        Tensor x = rand(N, N, 96L);
        int[] targets = randomTargets(N, N, 99L);
        FullState cpuState = fullState();
        FullState gpuState = copy(cpuState);
        bench("full train step (13 ops)",
                () -> fullTrainStep(cpu, x, targets, cpuState),
                () -> fullTrainStep(gpu, x, targets, gpuState).materialize());
    }

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

    private static Tensor mixedTen(TensorBackend backend, Tensor a, Tensor b) {
        Tensor value = backend.matmul(a, b);
        value = backend.gelu(value);
        value = backend.multiplyScalar(value, 0.5f);
        value = backend.subtract(value, b);
        value = backend.relu(value);
        value = backend.matmul(value, a);
        value = backend.sigmoid(value);
        value = backend.multiply(value, b);
        value = backend.tanh(value);
        return backend.neg(value);
    }

    private static Tensor mixedChain(TensorBackend backend, Tensor a, Tensor b, int operations) {
        Tensor value = backend.matmul(a, b);
        for (int index = 0; index < operations; index++) {
            value = switch (index % 5) {
                case 0 -> backend.gelu(value);
                case 1 -> backend.matmul(value, a);
                case 2 -> backend.add(value, b);
                case 3 -> backend.sigmoid(value);
                default -> backend.subtract(value, a);
            };
        }
        return value;
    }

    private static Tensor backward(TensorBackend backend, Tensor gradient, Tensor softmax,
                                   Tensor w1, Tensor w2, Tensor preActivation, Tensor offset) {
        Tensor value = backend.softmaxBackward(gradient, softmax);
        value = backend.matmul(value, w2);
        value = backend.geluBackward(preActivation, value);
        value = backend.matmul(value, w1);
        value = backend.subtract(value, offset);
        return backend.multiplyScalar(value, 0.5f);
    }

    private static MiniState miniState() {
        return new MiniState(
                rand(N, N, 92L), rand(N, N, 93L),
                rand(N, N, 94L), rand(N, N, 95L),
                cpu.zeros(N, N), cpu.zeros(N, N),
                cpu.zeros(N, N), cpu.zeros(N, N));
    }

    private static MiniState copy(MiniState source) {
        return new MiniState(
                clone(source.w1()), clone(source.w2()), clone(source.g1()), clone(source.g2()),
                clone(source.m1()), clone(source.v1()), clone(source.m2()), clone(source.v2()));
    }

    private static Tensor miniTrainStep(TensorBackend backend, Tensor input, MiniState state) {
        Tensor hidden = backend.gelu(backend.matmul(input, state.w1()));
        Tensor logits = backend.matmul(hidden, state.w2());
        backend.softmaxRows(logits);
        backend.multiplyScalar(logits, 0.1f);
        update(backend, state.w1(), state.g1(), state.m1(), state.v1());
        update(backend, state.w2(), state.g2(), state.m2(), state.v2());
        return state.w1();
    }

    private static FullState fullState() {
        return new FullState(
                rand(N, N, 97L), rand(N, N, 98L), rand(N, N, 100L),
                cpu.zeros(N, N), cpu.zeros(N, N),
                cpu.zeros(N, N), cpu.zeros(N, N));
    }

    private static FullState copy(FullState source) {
        return new FullState(
                clone(source.w1()), clone(source.w2()), clone(source.preActivation()),
                clone(source.m1()), clone(source.v1()), clone(source.m2()), clone(source.v2()));
    }

    private static Tensor fullTrainStep(TensorBackend backend, Tensor input, int[] targets,
                                        FullState state) {
        Tensor hidden = backend.gelu(backend.matmul(input, state.w1()));
        Tensor logits = backend.matmul(hidden, state.w2());
        Tensor dLogits = backend.crossEntropyGradient(logits, targets);
        Tensor dHidden = backend.matmul(dLogits, state.w2());
        dHidden = backend.geluBackward(state.preActivation(), dHidden);
        Tensor dInput = backend.matmul(dHidden, state.w1());
        update(backend, state.w1(), dInput, state.m1(), state.v1());
        update(backend, state.w2(), dLogits, state.m2(), state.v2());
        return state.w1();
    }

    private static void update(TensorBackend backend, Tensor weight, Tensor gradient,
                               Tensor moment, Tensor variance) {
        backend.adamWUpdate(weight, gradient, moment, variance,
                1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, 0.9f, 0.999f);
    }

    private record MiniState(Tensor w1, Tensor w2, Tensor g1, Tensor g2,
                             Tensor m1, Tensor v1, Tensor m2, Tensor v2) {}

    private record FullState(Tensor w1, Tensor w2, Tensor preActivation,
                             Tensor m1, Tensor v1, Tensor m2, Tensor v2) {}

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
