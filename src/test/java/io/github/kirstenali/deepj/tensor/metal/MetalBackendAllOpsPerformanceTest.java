package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.*;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

/**
 * CPU vs Metal GPU timing comparison for <b>all</b> TensorBackend operations.
 *
 * <p>Not a rigorous benchmark (use JMH for that). Disabled by default to keep CI stable.
 * Run manually with:
 * <pre>
 *   mvn test -Dtest=MetalBackendAllOpsPerformanceTest -Djunit.jupiter.conditions.deactivate=org.junit.jupiter.api.condition.* \
 *       -DskipTests=false -Dperf.size=512 -Dperf.iters.cpu=3 -Dperf.iters.gpu=10
 * </pre>
 */
@Disabled("Manual performance test; timings vary by machine and can be unstable in CI.")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public final class MetalBackendAllOpsPerformanceTest {

    private static TensorBackend cpu;
    private static TensorBackend gpu;
    private static TensorBackend previousBackend;

    private static int N;       // matrix dimension (N x N)
    private static int IT_CPU;
    private static int IT_GPU;

    /** Collects all results for a summary table printed at the end. */
    private static final Map<String, long[]> results = new LinkedHashMap<>();

    // ── setup / teardown ──────────────────────────────────────────

    @BeforeAll
    static void setUp() {
        Assumptions.assumeTrue(MetalNative.AVAILABLE, "Metal native library not available");
        cpu = new CpuBackend();
        // Force GPU dispatch for all sizes by setting thresholds to 0
        MetalBackend metal = new MetalBackend();
        gpu = metal;

        previousBackend = Tensor.backend();
        Tensor.setBackend(gpu);

        N      = intProp("perf.size", 512);
        IT_CPU = intProp("perf.iters.cpu", 3);
        IT_GPU = intProp("perf.iters.gpu", 10);

        System.out.println("\n╔══════════════════════════════════════════════════════════════════╗");
        System.out.println("║       DeepJ — CPU vs Metal GPU Performance (all ops)           ║");
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

    private static Tensor randPositive(int rows, int cols, long seed) {
        Tensor t = rand(rows, cols, seed);
        // Make all values positive (for log, sqrt, divide safety)
        for (int r = 0; r < t.rows; r++)
            for (int c = 0; c < t.cols; c++)
                t.data[r][c] = Math.abs(t.data[r][c]) + 0.01;
        return t;
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
        // warm-up
        for (int i = 0; i < 3; i++) { cpuOp.run(); gpuOp.run(); }

        long cpuNs = bestOfNanos(cpuOp, IT_CPU);
        long gpuNs = bestOfNanos(gpuOp, IT_GPU);
        results.put(label, new long[]{cpuNs, gpuNs});

        double cpuMs = cpuNs / 1_000_000.0;
        double gpuMs = gpuNs / 1_000_000.0;
        double speedup = cpuMs / gpuMs;
        String arrow = speedup >= 1.0 ? "🟢" : "🔴";
        System.out.printf("  %-28s CPU: %9.3f ms   GPU: %9.3f ms   speedup: %7.2fx  %s%n",
                label, cpuMs, gpuMs, speedup, arrow);
    }

    private static void printSummary() {
        System.out.println("\n┌────────────────────────────────┬────────────┬────────────┬──────────┐");
        System.out.println("│ Operation                      │   CPU (ms) │   GPU (ms) │ Speedup  │");
        System.out.println("├────────────────────────────────┼────────────┼────────────┼──────────┤");
        for (var entry : results.entrySet()) {
            double cpuMs = entry.getValue()[0] / 1_000_000.0;
            double gpuMs = entry.getValue()[1] / 1_000_000.0;
            System.out.printf("│ %-30s │ %10.3f │ %10.3f │ %7.2fx │%n",
                    entry.getKey(), cpuMs, gpuMs, cpuMs / gpuMs);
        }
        System.out.println("└────────────────────────────────┴────────────┴────────────┴──────────┘\n");
    }

    // ═══════════════════════════════════════════════════════════════
    //  MATMUL
    // ═══════════════════════════════════════════════════════════════

    @Test @Order(1)
    void matmul() {
        Tensor a = rand(N, N, 1L), b = rand(N, N, 2L);
        bench("matmul", () -> cpu.matmul(a, b), () -> gpu.matmul(a, b));
    }

    @Test @Order(2)
    void matmul_rectangular() {
        Tensor a = rand(N, N / 2, 3L), b = rand(N / 2, N, 4L);
        bench("matmul (rect)", () -> cpu.matmul(a, b), () -> gpu.matmul(a, b));
    }

    // ═══════════════════════════════════════════════════════════════
    //  ELEMENT-WISE BINARY
    // ═══════════════════════════════════════════════════════════════

    @Test @Order(10)
    void add() {
        Tensor a = rand(N, N, 5L), b = rand(N, N, 6L);
        bench("add", () -> cpu.add(a, b), () -> gpu.add(a, b));
    }

    @Test @Order(11)
    void subtract() {
        Tensor a = rand(N, N, 7L), b = rand(N, N, 8L);
        bench("subtract", () -> cpu.subtract(a, b), () -> gpu.subtract(a, b));
    }

    @Test @Order(12)
    void multiply() {
        Tensor a = rand(N, N, 9L), b = rand(N, N, 10L);
        bench("multiply", () -> cpu.multiply(a, b), () -> gpu.multiply(a, b));
    }

    @Test @Order(13)
    void divide() {
        Tensor a = rand(N, N, 11L), b = randPositive(N, N, 12L);
        bench("divide", () -> cpu.divide(a, b), () -> gpu.divide(a, b));
    }

    // ═══════════════════════════════════════════════════════════════
    //  SCALAR OPS
    // ═══════════════════════════════════════════════════════════════

    @Test @Order(20)
    void multiplyScalar() {
        Tensor a = rand(N, N, 13L);
        bench("multiplyScalar", () -> cpu.multiplyScalar(a, 2.5), () -> gpu.multiplyScalar(a, 2.5));
    }

    @Test @Order(21)
    void addScalar() {
        Tensor a = rand(N, N, 14L);
        bench("addScalar", () -> cpu.addScalar(a, 1.0), () -> gpu.addScalar(a, 1.0));
    }

    @Test @Order(22)
    void divideScalar() {
        Tensor a = rand(N, N, 15L);
        bench("divideScalar", () -> cpu.divideScalar(a, 3.0), () -> gpu.divideScalar(a, 3.0));
    }

    // ═══════════════════════════════════════════════════════════════
    //  BROADCAST OPS
    // ═══════════════════════════════════════════════════════════════

    @Test @Order(30)
    void addRowVector() {
        Tensor a = rand(N, N, 16L), v = rand(1, N, 17L);
        bench("addRowVector", () -> cpu.addRowVector(a, v), () -> gpu.addRowVector(a, v));
    }

    @Test @Order(31)
    void addBroadcastCols() {
        Tensor a = rand(N, N, 18L), v = rand(N, 1, 19L);
        bench("addBroadcastCols", () -> cpu.addBroadcastCols(a, v), () -> gpu.addBroadcastCols(a, v));
    }

    @Test @Order(32)
    void subtractBroadcastCols() {
        Tensor a = rand(N, N, 20L), v = rand(N, 1, 21L);
        bench("subtractBroadcastCols", () -> cpu.subtractBroadcastCols(a, v), () -> gpu.subtractBroadcastCols(a, v));
    }

    @Test @Order(33)
    void multiplyBroadcastCols() {
        Tensor a = rand(N, N, 22L), v = rand(N, 1, 23L);
        bench("multiplyBroadcastCols", () -> cpu.multiplyBroadcastCols(a, v), () -> gpu.multiplyBroadcastCols(a, v));
    }

    @Test @Order(34)
    void divideBroadcastCols() {
        Tensor a = rand(N, N, 24L), v = randPositive(N, 1, 25L);
        bench("divideBroadcastCols", () -> cpu.divideBroadcastCols(a, v), () -> gpu.divideBroadcastCols(a, v));
    }

    @Test @Order(35)
    void addBroadcastRows() {
        Tensor a = rand(N, N, 26L), v = rand(1, N, 27L);
        bench("addBroadcastRows", () -> cpu.addBroadcastRows(a, v), () -> gpu.addBroadcastRows(a, v));
    }

    @Test @Order(36)
    void multiplyBroadcastRows() {
        Tensor a = rand(N, N, 28L), v = rand(1, N, 29L);
        bench("multiplyBroadcastRows", () -> cpu.multiplyBroadcastRows(a, v), () -> gpu.multiplyBroadcastRows(a, v));
    }

    // ═══════════════════════════════════════════════════════════════
    //  REDUCTIONS
    // ═══════════════════════════════════════════════════════════════

    @Test @Order(40)
    void sumRows() {
        Tensor a = rand(N, N, 30L);
        bench("sumRows", () -> cpu.sumRows(a), () -> gpu.sumRows(a));
    }

    @Test @Order(41)
    void sumAlongRows() {
        Tensor a = rand(N, N, 31L);
        bench("sumAlongRows", () -> cpu.sumAlongRows(a), () -> gpu.sumAlongRows(a));
    }

    @Test @Order(42)
    void sumAlongCols() {
        Tensor a = rand(N, N, 32L);
        bench("sumAlongCols", () -> cpu.sumAlongCols(a), () -> gpu.sumAlongCols(a));
    }

    @Test @Order(43)
    void meanAlongRows() {
        Tensor a = rand(N, N, 33L);
        bench("meanAlongRows", () -> cpu.meanAlongRows(a), () -> gpu.meanAlongRows(a));
    }

    @Test @Order(44)
    void varianceAlongRows() {
        Tensor a = rand(N, N, 34L);
        bench("varianceAlongRows", () -> cpu.varianceAlongRows(a), () -> gpu.varianceAlongRows(a));
    }

    @Test @Order(45)
    void maxAlongRows() {
        Tensor a = rand(N, N, 35L);
        bench("maxAlongRows", () -> cpu.maxAlongRows(a), () -> gpu.maxAlongRows(a));
    }

    @Test @Order(46)
    void sum() {
        Tensor a = rand(N, N, 36L);
        bench("sum (scalar)", () -> cpu.sum(a), () -> gpu.sum(a));
    }

    @Test @Order(47)
    void sumAbs() {
        Tensor a = rand(N, N, 37L);
        bench("sumAbs (scalar)", () -> cpu.sumAbs(a), () -> gpu.sumAbs(a));
    }

    // ═══════════════════════════════════════════════════════════════
    //  UNARY MATH
    // ═══════════════════════════════════════════════════════════════

    @Test @Order(50)
    void sqrt() {
        Tensor a = randPositive(N, N, 38L);
        bench("sqrt", () -> cpu.sqrt(a), () -> gpu.sqrt(a));
    }

    @Test @Order(51)
    void pow() {
        Tensor a = randPositive(N, N, 39L);
        bench("pow (^2)", () -> cpu.pow(a, 2.0), () -> gpu.pow(a, 2.0));
    }

    @Test @Order(52)
    void neg() {
        Tensor a = rand(N, N, 40L);
        bench("neg", () -> cpu.neg(a), () -> gpu.neg(a));
    }

    @Test @Order(53)
    void exp() {
        Tensor a = rand(N, N, 41L);
        bench("exp", () -> cpu.exp(a), () -> gpu.exp(a));
    }

    @Test @Order(54)
    void log() {
        Tensor a = randPositive(N, N, 42L);
        bench("log", () -> cpu.log(a), () -> gpu.log(a));
    }

    @Test @Order(55)
    void clamp() {
        Tensor a = rand(N, N, 43L);
        bench("clamp", () -> cpu.clamp(a, -0.5, 0.5), () -> gpu.clamp(a, -0.5, 0.5));
    }

    @Test @Order(56)
    void transpose() {
        Tensor a = rand(N, N, 44L);
        bench("transpose", () -> cpu.transpose(a), () -> gpu.transpose(a));
    }

    // ═══════════════════════════════════════════════════════════════
    //  ACTIVATIONS
    // ═══════════════════════════════════════════════════════════════

    @Test @Order(60)
    void tanh() {
        Tensor a = rand(N, N, 45L);
        bench("tanh", () -> cpu.tanh(a), () -> gpu.tanh(a));
    }

    @Test @Order(61)
    void sigmoid() {
        Tensor a = rand(N, N, 46L);
        bench("sigmoid", () -> cpu.sigmoid(a), () -> gpu.sigmoid(a));
    }

    @Test @Order(62)
    void relu() {
        Tensor a = rand(N, N, 47L);
        bench("relu", () -> cpu.relu(a), () -> gpu.relu(a));
    }

    @Test @Order(63)
    void reluBackward() {
        Tensor input = rand(N, N, 48L), grad = rand(N, N, 49L);
        bench("reluBackward", () -> cpu.reluBackward(input, grad), () -> gpu.reluBackward(input, grad));
    }

    @Test @Order(64)
    void gelu() {
        Tensor a = rand(N, N, 50L);
        bench("gelu", () -> cpu.gelu(a), () -> gpu.gelu(a));
    }

    @Test @Order(65)
    void geluBackward() {
        Tensor input = rand(N, N, 51L), grad = rand(N, N, 52L);
        bench("geluBackward", () -> cpu.geluBackward(input, grad), () -> gpu.geluBackward(input, grad));
    }

    // ═══════════════════════════════════════════════════════════════
    //  SOFTMAX
    // ═══════════════════════════════════════════════════════════════

    @Test @Order(70)
    void softmaxRows() {
        Tensor a = rand(N, N, 53L);
        bench("softmaxRows", () -> cpu.softmaxRows(a), () -> gpu.softmaxRows(a));
    }

    @Test @Order(71)
    void softmaxBackward() {
        Tensor grad = rand(N, N, 54L);
        Tensor smOut = cpu.softmaxRows(rand(N, N, 55L));
        bench("softmaxBackward", () -> cpu.softmaxBackward(grad, smOut), () -> gpu.softmaxBackward(grad, smOut));
    }

    // ═══════════════════════════════════════════════════════════════
    //  FUSED / HIGH-LEVEL OPS
    // ═══════════════════════════════════════════════════════════════

    @Test @Order(80)
    void crossEntropyLoss() {
        int vocabSize = N;
        int batchSize = Math.min(N, 128);
        Tensor logits = rand(batchSize, vocabSize, 56L);
        int[] targets = randomTargets(batchSize, vocabSize, 57L);
        bench("crossEntropyLoss", () -> cpu.crossEntropyLoss(logits, targets), () -> gpu.crossEntropyLoss(logits, targets));
    }

    @Test @Order(81)
    void crossEntropyGradient() {
        int vocabSize = N;
        int batchSize = Math.min(N, 128);
        Tensor logits = rand(batchSize, vocabSize, 58L);
        int[] targets = randomTargets(batchSize, vocabSize, 59L);
        bench("crossEntropyGradient", () -> cpu.crossEntropyGradient(logits, targets), () -> gpu.crossEntropyGradient(logits, targets));
    }

    @Test @Order(82)
    void adamWUpdate() {
        Tensor w  = rand(N, N, 60L);
        Tensor g  = rand(N, N, 61L);
        Tensor mt = cpu.zeros(N, N);
        Tensor vt = cpu.zeros(N, N);

        // Clone for GPU so they start from the same state
        Tensor wG  = clone(w);
        Tensor gG  = clone(g);
        Tensor mtG = cpu.zeros(N, N);
        Tensor vtG = cpu.zeros(N, N);

        bench("adamWUpdate",
                () -> cpu.adamWUpdate(w, g, mt, vt, 1e-3, 0.9, 0.999, 1e-8, 0.01, 0.9, 0.999),
                () -> gpu.adamWUpdate(wG, gG, mtG, vtG, 1e-3, 0.9, 0.999, 1e-8, 0.01, 0.9, 0.999));
    }

    @Test @Order(83)
    void layerNormBackward() {
        int dim = N;
        Tensor dXHat = rand(N, dim, 62L);
        Tensor xHat  = rand(N, dim, 63L);
        Tensor std   = randPositive(N, 1, 64L);
        bench("layerNormBackward", () -> cpu.layerNormBackward(dXHat, xHat, std, dim), () -> gpu.layerNormBackward(dXHat, xHat, std, dim));
    }

    // ═══════════════════════════════════════════════════════════════
    //  DATA ACCESSORS / INDEXING
    // ═══════════════════════════════════════════════════════════════

    @Test @Order(90)
    void sliceRows() {
        int vocabSize = N;
        Tensor embeddings = rand(vocabSize, N, 65L);
        int[] indices = randomTargets(Math.min(128, N), vocabSize, 66L);
        bench("sliceRows", () -> cpu.sliceRows(embeddings, indices, N), () -> gpu.sliceRows(embeddings, indices, N));
    }

    @Test @Order(91)
    void scatterAddRows() {
        int vocabSize = N;
        int seqLen = Math.min(128, N);
        Tensor target1 = cpu.zeros(vocabSize, N);
        Tensor target2 = cpu.zeros(vocabSize, N);
        Tensor grad = rand(seqLen, N, 67L);
        int[] indices = randomTargets(seqLen, vocabSize, 68L);
        bench("scatterAddRows",
                () -> cpu.scatterAddRows(target1, indices, grad),
                () -> gpu.scatterAddRows(target2, indices, grad));
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
        Tensor c = new Tensor(t.rows, t.cols);
        for (int r = 0; r < t.rows; r++)
            System.arraycopy(t.data[r], 0, c.data[r], 0, t.cols);
        return c;
    }
}

