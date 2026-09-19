package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Random;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.*;

public final class MetalBackendTest {

    private static final float ADAM_LR = 1e-3f;
    private static final float ADAM_BETA1 = 0.9f;
    private static final float ADAM_BETA2 = 0.999f;
    private static final float ADAM_EPSILON = 1e-8f;
    private static final float ADAM_WEIGHT_DECAY = 0.01f;
    private static CpuBackend cpu;
    private static TensorBackend gpu;
    private static TensorBackend previousBackend;

    @BeforeAll
    static void setUp() {
        Assumptions.assumeTrue(MetalBackend.isAvailable(), "Metal device not available");
        cpu = new CpuBackend();
        gpu = new MetalBackend();
        previousBackend = Tensor.backend();
        Tensor.setBackend(gpu);
    }

    @AfterAll
    static void tearDown() {
        if (previousBackend != null) {
            Tensor.setBackend(previousBackend);
        }
    }

    private static Tensor randomTensor(int rows, int cols, long seed) {
        return cpu.random(rows, cols, new Random(seed));
    }

    @Test
    void metalNativeIsAvailable() {
        assertTrue(MetalBackend.isAvailable());
    }

    @Test
    void smallMatmulMatchesExpectedValues() {
        Tensor a = Tensor.from2D(new float[][]{
                {1.0f, 2.0f, 3.0f},
                {4.0f, 5.0f, 6.0f}
        });

        Tensor b = Tensor.from2D(new float[][]{
                {7.0f, 8.0f},
                {9.0f, 10.0f},
                {11.0f, 12.0f}
        });

        Tensor c = gpu.matmul(a, b);
        c.materialize();
        assertEquals(2, c.rows);
        assertEquals(2, c.cols);
        assertEquals(58.0f,  c.data[0], 1e-6f);
        assertEquals(64.0f,  c.data[1], 1e-6f);
        assertEquals(139.0f, c.data[1 * 2 + 0], 1e-6f);
        assertEquals(154.0f, c.data[1 * 2 + 1], 1e-6f);
    }

    @Test
    void matmulMatchesCpu_rectangular() {
        Tensor a = randomTensor(128, 192, 1L);
        Tensor b = randomTensor(192, 64, 2L);

        Tensor expected = cpu.matmul(a, b);
        Tensor actual = gpu.matmul(a, b);
        assertTensorClose(expected, actual, 1e-4f, 1e-4f);
    }

    @Test
    void matmulRejectsShapeMismatch() {
        Tensor a = randomTensor(3, 4, 3L);
        Tensor b = randomTensor(5, 2, 4L);

        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> gpu.matmul(a, b));
        assertTrue(ex.getMessage().contains("Shape mismatch"));
    }

    @Test
    void broadcastAndScalarOpsMatchCpu() {
        withGpuBackend(MetalBackendTest::assertBroadcastAndScalarOps);
    }

    @Test
    void reductionsAndTransposeMatchCpu() {
        MetalBackend gpuBackend = new MetalBackend();
        TensorBackend oldBackend = Tensor.backend();
        Tensor.setBackend(gpuBackend);

        try {
            Tensor a = randomTensor(32, 24, 111L);

            assertTensorClose(cpu.transpose(a), gpuBackend.transpose(a), 1e-4f, 1e-4f);
            assertTensorClose(cpu.sumRows(a), gpuBackend.sumRows(a), 1e-4f, 1e-4f);
            assertTensorClose(cpu.sumAlongRows(a), gpuBackend.sumAlongRows(a), 1e-4f, 1e-4f);
            assertTensorClose(cpu.sumAlongCols(a), gpuBackend.sumAlongCols(a), 1e-4f, 1e-4f);
            assertTensorClose(cpu.meanAlongRows(a), gpuBackend.meanAlongRows(a), 1e-4f, 1e-4f);
            assertTensorClose(cpu.varianceAlongRows(a), gpuBackend.varianceAlongRows(a), 1e-4f, 1e-4f);
            assertEquals(cpu.sum(a), gpuBackend.sum(a), 1e-4f);
        } finally {
            gpuBackend.releaseResources();
            Tensor.setBackend(oldBackend);
        }
    }

    @Test
    void maxClampPowAndScatterAddRowsMatchCpu() {
        withGpuBackend(MetalBackendTest::assertMaxClampPowAndScatters);
    }

    @Test
    void scatterAddRowsDuplicateIndicesAtomicModeMatchesCpu() {
        withGpuBackend(MetalBackendTest::assertAtomicScatter);
    }

    @Test
    void scalarSumAbsAndCrossEntropyLossMatchCpu() {
        withGpuBackend(MetalBackendTest::assertScalarLosses);
    }

    @Test
    void softmaxBackwardMatchesCpu() {
        withGpuBackend(MetalBackendTest::assertSoftmaxBackward);
    }

    @Test
    void softmaxBackwardRejectsShapeMismatch() {
        Tensor gradOutput = randomTensor(2, 3, 21L);
        Tensor softmaxOut = randomTensor(3, 2, 22L);

        IllegalArgumentException ex = assertThrows(
                IllegalArgumentException.class,
                () -> gpu.softmaxBackward(gradOutput, softmaxOut));
        assertTrue(ex.getMessage().contains("Shape mismatch"));
    }

    @Test
    void layerNormBackwardMatchesCpu() {
        withGpuBackend(MetalBackendTest::assertLayerNormBackward);
    }

    @Test
    void layerNormBackwardRejectsStdShapeMismatch() {
        Tensor dXHat = randomTensor(4, 8, 41L);
        Tensor xHat = randomTensor(4, 8, 42L);
        Tensor badStd = randomTensor(4, 2, 43L);

        IllegalArgumentException ex = assertThrows(
                IllegalArgumentException.class,
                () -> gpu.layerNormBackward(dXHat, xHat, badStd, dXHat.cols));
        assertTrue(ex.getMessage().contains("layerNormBackward: std"));
    }

    @Test
    void crossEntropyGradientMatchesCpu() {
        withGpuBackend(MetalBackendTest::assertCrossEntropyGradient);
    }

    @Test
    void crossEntropyGradientRejectsTargetLengthMismatch() {
        Tensor logits = randomTensor(4, 8, 61L);
        int[] badTargets = new int[]{0, 1, 2};

        IllegalArgumentException ex = assertThrows(
                IllegalArgumentException.class,
                () -> gpu.crossEntropyGradient(logits, badTargets));
        assertTrue(ex.getMessage().contains("targets length"));
    }

    @Test
    void adamWUpdateMatchesCpu_singleStep() {
        withGpuBackend(MetalBackendTest::assertSingleAdamStep);
    }

    @Test
    void adamWUpdateMatchesCpu_multipleSteps() {
        withGpuBackend(MetalBackendTest::assertMultipleAdamSteps);
    }

    private static void withGpuBackend(Consumer<MetalBackend> assertion) {
        MetalBackend backend = new MetalBackend();
        TensorBackend oldBackend = Tensor.backend();
        Tensor.setBackend(backend);
        try {
            assertion.accept(backend);
        } finally {
            backend.releaseResources();
            Tensor.setBackend(oldBackend);
        }
    }

    private static void assertBroadcastAndScalarOps(MetalBackend backend) {
        Tensor a = randomTensor(8, 16, 100L);
        Tensor row = randomTensor(1, 16, 101L);
        Tensor col = randomTensor(8, 1, 102L);
        assertTensorClose(cpu.addRowVector(a, row), backend.addRowVector(a, row), 1e-4f, 1e-4f);
        assertTensorClose(cpu.addBroadcastCols(a, col), backend.addBroadcastCols(a, col), 1e-4f, 1e-4f);
        assertTensorClose(cpu.subtractBroadcastCols(a, col), backend.subtractBroadcastCols(a, col), 1e-4f, 1e-4f);
        assertTensorClose(cpu.divideBroadcastCols(a, col), backend.divideBroadcastCols(a, col), 1e-4f, 1e-4f);
        assertTensorClose(cpu.multiplyBroadcastCols(a, col), backend.multiplyBroadcastCols(a, col), 1e-4f, 1e-4f);
        assertTensorClose(cpu.multiplyBroadcastRows(a, row), backend.multiplyBroadcastRows(a, row), 1e-4f, 1e-4f);
        assertTensorClose(cpu.addScalar(a, 0.25f), backend.addScalar(a, 0.25f), 1e-4f, 1e-4f);
        assertTensorClose(cpu.divideScalar(a, 1.5f), backend.divideScalar(a, 1.5f), 1e-4f, 1e-4f);
    }

    private static void assertMaxClampPowAndScatters(MetalBackend backend) {
        assertMaxClampAndPow(backend);
        assertScatterRows(122L, 123L, new int[]{3, 1, 7, 2, 4, 5}, 1e-4f);
        assertScatterRows(124L, 125L, new int[]{3, 1, 7, 1, 4, 3}, 1e-3f);
    }

    private static void assertMaxClampAndPow(MetalBackend backend) {
        Tensor a = randomTensor(16, 20, 121L);
        assertTensorClose(cpu.maxAlongRows(a), backend.maxAlongRows(a), 1e-4f, 1e-4f);
        assertTensorClose(cpu.clamp(a, -0.25f, 0.35f), backend.clamp(a, -0.25f, 0.35f), 1e-4f, 1e-4f);
        assertTensorClose(cpu.pow(a, 2.0f), backend.pow(a, 2.0f), 1e-4f, 1e-4f);
    }

    private static void assertScatterRows(long targetSeed, long gradientSeed,
                                          int[] indices, float tolerance) {
        Tensor targetCpu = randomTensor(10, 8, targetSeed);
        Tensor targetGpu = new Tensor(targetCpu);
        Tensor gradient = randomTensor(indices.length, 8, gradientSeed);
        cpu.scatterAddRows(targetCpu, indices, gradient);
        Tensor.scatterAddRows(targetGpu, indices, gradient);
        assertTensorClose(targetCpu, targetGpu, tolerance, tolerance);
    }

    private static void assertAtomicScatter(MetalBackend backend) {
        Tensor targetCpu = randomTensor(64, 32, 129L);
        Tensor targetGpu = new Tensor(targetCpu);
        Tensor gradient = randomTensor(128, 32, 130L);
        int[] indices = randomIndices(128, targetCpu.rows, 131L);
        cpu.scatterAddRows(targetCpu, indices, gradient);
        Tensor.scatterAddRows(targetGpu, indices, gradient);
        assertTensorClose(targetCpu, targetGpu, 1e-3f, 1e-3f);
    }

    private static int[] randomIndices(int length, int bound, long seed) {
        int[] indices = new int[length];
        Random random = new Random(seed);
        for (int index = 0; index < length; index++) indices[index] = random.nextInt(bound);
        return indices;
    }

    private static void assertScalarLosses(MetalBackend backend) {
        assertSumAbs(backend, 23, 17, 126L, 1e-4f);
        assertCrossEntropyLoss(backend, 19, 31, 127L, 128L, 1e-4f);
        assertSumAbs(backend, 29, 513, 132L, 1e-3f);
        assertCrossEntropyLoss(backend, 23, 777, 133L, 134L, 1e-3f);
    }

    private static void assertSumAbs(MetalBackend backend, int rows, int columns,
                                     long seed, float tolerance) {
        Tensor value = randomTensor(rows, columns, seed);
        assertEquals(cpu.sumAbs(value), backend.sumAbs(value), tolerance);
    }

    private static void assertCrossEntropyLoss(MetalBackend backend, int rows, int columns,
                                               long valueSeed, long targetSeed, float tolerance) {
        Tensor logits = randomTensor(rows, columns, valueSeed);
        int[] targets = randomTargets(rows, columns, targetSeed);
        assertEquals(cpu.crossEntropyLoss(logits, targets),
                backend.crossEntropyLoss(logits, targets), tolerance);
    }

    private static int[] randomTargets(int rows, int columns, long seed) {
        int[] targets = new int[rows];
        Random random = new Random(seed);
        for (int row = 0; row < rows; row++) targets[row] = random.nextInt(columns);
        return targets;
    }

    private static void assertSoftmaxBackward(MetalBackend backend) {
        assertSoftmaxBackward(backend, 32, 64, 10L, 11L);
        assertSoftmaxBackward(backend, 17, 769, 12L, 13L);
    }

    private static void assertSoftmaxBackward(MetalBackend backend, int rows, int columns,
                                              long gradientSeed, long logitsSeed) {
        Tensor gradient = randomTensor(rows, columns, gradientSeed);
        Tensor softmax = cpu.softmaxRows(randomTensor(rows, columns, logitsSeed));
        Tensor expected = cpu.softmaxBackward(gradient, softmax);
        Tensor actual = backend.softmaxBackward(gradient, softmax);
        assertTensorClose(expected, actual, 1e-4f, 1e-4f);
    }

    private static void assertLayerNormBackward(MetalBackend backend) {
        assertLayerNormBackward(backend, 16, 32, 31L, 32L);
        assertLayerNormBackward(backend, 11, 777, 33L, 34L);
    }

    private static void assertLayerNormBackward(MetalBackend backend, int rows, int columns,
                                                long gradientSeed, long inputSeed) {
        Tensor gradient = randomTensor(rows, columns, gradientSeed);
        Tensor normalized = randomTensor(rows, columns, inputSeed);
        Tensor deviation = standardDeviations(normalized);
        Tensor expected = cpu.layerNormBackward(gradient, normalized, deviation, columns);
        Tensor actual = backend.layerNormBackward(gradient, normalized, deviation, columns);
        assertTensorClose(expected, actual, 1e-4f, 1e-4f);
    }

    private static Tensor standardDeviations(Tensor normalized) {
        Tensor deviation = new Tensor(normalized.rows, 1);
        for (int row = 0; row < deviation.rows; row++) {
            deviation.data[row] = 0.5f + Math.abs(normalized.data[row * normalized.cols]);
        }
        return deviation;
    }

    private static void assertCrossEntropyGradient(MetalBackend backend) {
        assertCrossEntropyGradient(backend, 32, 64, 51L, 52L);
        assertCrossEntropyGradient(backend, 17, 769, 53L, 54L);
    }

    private static void assertCrossEntropyGradient(MetalBackend backend, int rows, int columns,
                                                   long valueSeed, long targetSeed) {
        Tensor logits = randomTensor(rows, columns, valueSeed);
        int[] targets = randomTargets(rows, columns, targetSeed);
        Tensor expected = cpu.crossEntropyGradient(logits, targets);
        Tensor actual = backend.crossEntropyGradient(logits, targets);
        assertTensorClose(expected, actual, 1e-3f, 1e-2f);
    }

    private static void assertSingleAdamStep(MetalBackend backend) {
        AdamState expected = adamState(32, 64, 71L);
        AdamState actual = copyState(expected);
        Tensor gradient = randomTensor(32, 64, 72L);
        updateAdam(cpu, expected, gradient, 1);
        updateAdam(backend, actual, new Tensor(gradient), 1);
        assertAdamState(expected, actual);
    }

    private static void assertMultipleAdamSteps(MetalBackend backend) {
        AdamState expected = adamState(16, 48, 81L);
        AdamState actual = copyState(expected);
        for (int step = 1; step <= 5; step++) {
            Tensor gradient = randomTensor(16, 48, 90L + step);
            updateAdam(cpu, expected, gradient, step);
            updateAdam(backend, actual, new Tensor(gradient), step);
        }
        assertAdamState(expected, actual);
    }

    private static AdamState adamState(int rows, int columns, long seed) {
        return new AdamState(
                randomTensor(rows, columns, seed), new Tensor(rows, columns),
                new Tensor(rows, columns));
    }

    private static AdamState copyState(AdamState source) {
        return new AdamState(new Tensor(source.weights()), new Tensor(source.firstMoment()),
                new Tensor(source.secondMoment()));
    }

    private static void updateAdam(TensorBackend backend, AdamState state,
                                   Tensor gradient, int step) {
        float correction1 = 1.0f - (float) Math.pow(ADAM_BETA1, step);
        float correction2 = 1.0f - (float) Math.pow(ADAM_BETA2, step);
        backend.adamWUpdate(state.weights(), gradient, state.firstMoment(), state.secondMoment(),
                ADAM_LR, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, ADAM_WEIGHT_DECAY,
                correction1, correction2);
    }

    private static void assertAdamState(AdamState expected, AdamState actual) {
        assertTensorClose(expected.weights(), actual.weights(), 1e-4f, 1e-4f);
        assertTensorClose(expected.firstMoment(), actual.firstMoment(), 1e-4f, 1e-4f);
        assertTensorClose(expected.secondMoment(), actual.secondMoment(), 1e-4f, 1e-4f);
    }

    private record AdamState(Tensor weights, Tensor firstMoment, Tensor secondMoment) {}

    private static void assertTensorClose(Tensor expected, Tensor actual, double atol, double rtol) {
        assertEquals(expected.rows, actual.rows, "rows mismatch");
        assertEquals(expected.cols, actual.cols, "cols mismatch");
        expected.materialize();
        actual.materialize();

        for (int r = 0; r < expected.rows; r++) {
            for (int c = 0; c < expected.cols; c++) {
                double e = expected.data[r * expected.cols + c];
                double a = actual.data[r * actual.cols + c];
                double tol = atol + rtol * Math.abs(e);
                assertTrue(Math.abs(e - a) <= tol, "Mismatch at (" + r + "," + c + ") expected=" + e + " actual=" + a);
            }
        }
    }
}
