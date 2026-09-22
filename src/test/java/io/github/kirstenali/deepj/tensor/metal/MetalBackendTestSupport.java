package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Random;
import java.util.List;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.*;

abstract class MetalBackendTestSupport {

    protected static final float ADAM_LR = 1e-3f;
    protected static final float ADAM_BETA1 = 0.9f;
    protected static final float ADAM_BETA2 = 0.999f;
    protected static final float ADAM_EPSILON = 1e-8f;
    protected static final float ADAM_WEIGHT_DECAY = 0.01f;
    protected static CpuBackend cpu;
    protected static TensorBackend gpu;
    protected static TensorBackend previousBackend;

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
        if (gpu != null) gpu.releaseResources();
        if (previousBackend != null) {
            Tensor.setBackend(previousBackend);
        }
    }

    protected static Tensor randomTensor(int rows, int cols, long seed) {
        return cpu.random(rows, cols, new Random(seed));
    }

    protected static void withGpuBackend(Consumer<MetalBackend> assertion) {
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

    protected static void assertBroadcastAndScalarOps(MetalBackend backend) {
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

    protected static void assertMaxClampPowAndScatters(MetalBackend backend) {
        assertMaxClampAndPow(backend);
        assertScatterRows(122L, 123L, new int[]{3, 1, 7, 2, 4, 5}, 1e-4f);
        assertScatterRows(124L, 125L, new int[]{3, 1, 7, 1, 4, 3}, 1e-3f);
    }

    protected static void assertMaxClampAndPow(MetalBackend backend) {
        Tensor a = randomTensor(16, 20, 121L);
        assertTensorClose(cpu.maxAlongRows(a), backend.maxAlongRows(a), 1e-4f, 1e-4f);
        assertTensorClose(cpu.clamp(a, -0.25f, 0.35f), backend.clamp(a, -0.25f, 0.35f), 1e-4f, 1e-4f);
        assertTensorClose(cpu.pow(a, 2.0f), backend.pow(a, 2.0f), 1e-4f, 1e-4f);
    }

    protected static void assertGlobalL2Norm(MetalBackend backend) {
        List<Tensor> tensors = List.of(randomTensor(7, 11, 126L),
                randomTensor(3, 5, 127L));
        assertEquals(cpu.l2Norm(tensors), backend.l2Norm(tensors), 1e-4f);
    }

    protected static void assertTemporaryRelease(MetalBackend backend) {
        Tensor retained = randomTensor(4, 4, 128L).retainDeviceBuffer();
        Tensor temporary = backend.neg(retained);
        temporary.materialize();
        Object retainedTag = retained.getGpuTag();
        backend.releaseTemporaryResources();
        assertSame(retainedTag, retained.getGpuTag());
        assertNull(temporary.getGpuTag());
        assertTensorClose(cpu.neg(retained), backend.neg(retained), 1e-6f, 1e-6f);
    }

    protected static void assertScatterRows(long targetSeed, long gradientSeed,
                                          int[] indices, float tolerance) {
        Tensor targetCpu = randomTensor(10, 8, targetSeed);
        Tensor targetGpu = new Tensor(targetCpu);
        Tensor gradient = randomTensor(indices.length, 8, gradientSeed);
        cpu.scatterAddRows(targetCpu, indices, gradient);
        Tensor.scatterAddRows(targetGpu, indices, gradient);
        assertTensorClose(targetCpu, targetGpu, tolerance, tolerance);
    }

    protected static void assertAtomicScatter(MetalBackend backend) {
        Tensor targetCpu = randomTensor(64, 32, 129L);
        Tensor targetGpu = new Tensor(targetCpu);
        Tensor gradient = randomTensor(128, 32, 130L);
        int[] indices = randomIndices(128, targetCpu.rows, 131L);
        cpu.scatterAddRows(targetCpu, indices, gradient);
        Tensor.scatterAddRows(targetGpu, indices, gradient);
        assertTensorClose(targetCpu, targetGpu, 1e-3f, 1e-3f);
    }

    protected static int[] randomIndices(int length, int bound, long seed) {
        int[] indices = new int[length];
        Random random = new Random(seed);
        for (int index = 0; index < length; index++) {
            indices[index] = random.nextInt(bound);
        }
        return indices;
    }

    protected static void assertScalarLosses(MetalBackend backend) {
        assertSumAbs(backend, 23, 17, 126L, 1e-4f);
        assertCrossEntropyLoss(backend, 19, 31, 127L, 128L, 1e-4f);
        assertSumAbs(backend, 29, 513, 132L, 1e-3f);
        assertCrossEntropyLoss(backend, 23, 777, 133L, 134L, 1e-3f);
    }

    protected static void assertSumAbs(MetalBackend backend, int rows, int columns,
                                     long seed, float tolerance) {
        Tensor value = randomTensor(rows, columns, seed);
        assertEquals(cpu.sumAbs(value), backend.sumAbs(value), tolerance);
    }

    protected static void assertCrossEntropyLoss(MetalBackend backend, int rows, int columns,
                                               long valueSeed, long targetSeed, float tolerance) {
        Tensor logits = randomTensor(rows, columns, valueSeed);
        int[] targets = randomTargets(rows, columns, targetSeed);
        assertEquals(cpu.crossEntropyLoss(logits, targets),
                backend.crossEntropyLoss(logits, targets), tolerance);
    }

    protected static int[] randomTargets(int rows, int columns, long seed) {
        int[] targets = new int[rows];
        Random random = new Random(seed);
        for (int row = 0; row < rows; row++) {
            targets[row] = random.nextInt(columns);
        }
        return targets;
    }

    protected static void assertSoftmaxBackward(MetalBackend backend) {
        assertSoftmaxBackward(backend, 32, 64, 10L, 11L);
        assertSoftmaxBackward(backend, 17, 769, 12L, 13L);
    }

    protected static void assertSoftmaxBackward(MetalBackend backend, int rows, int columns,
                                              long gradientSeed, long logitsSeed) {
        Tensor gradient = randomTensor(rows, columns, gradientSeed);
        Tensor softmax = cpu.softmaxRows(randomTensor(rows, columns, logitsSeed));
        Tensor expected = cpu.softmaxBackward(gradient, softmax);
        Tensor actual = backend.softmaxBackward(gradient, softmax);
        assertTensorClose(expected, actual, 1e-4f, 1e-4f);
    }

    protected static void assertLayerNormBackward(MetalBackend backend) {
        assertLayerNormBackward(backend, 16, 32, 31L, 32L);
        assertLayerNormBackward(backend, 11, 777, 33L, 34L);
    }

    protected static void assertLayerNormBackward(MetalBackend backend, int rows, int columns,
                                                long gradientSeed, long inputSeed) {
        Tensor gradient = randomTensor(rows, columns, gradientSeed);
        Tensor normalized = randomTensor(rows, columns, inputSeed);
        Tensor deviation = standardDeviations(normalized);
        Tensor expected = cpu.layerNormBackward(gradient, normalized, deviation, columns);
        Tensor actual = backend.layerNormBackward(gradient, normalized, deviation, columns);
        assertTensorClose(expected, actual, 1e-4f, 1e-4f);
    }

    protected static Tensor standardDeviations(Tensor normalized) {
        Tensor deviation = new Tensor(normalized.rows, 1);
        for (int row = 0; row < deviation.rows; row++) {
            deviation.data[row] = 0.5f + Math.abs(normalized.data[row * normalized.cols]);
        }
        return deviation;
    }

    protected static void assertCrossEntropyGradient(MetalBackend backend) {
        assertCrossEntropyGradient(backend, 32, 64, 51L, 52L);
        assertCrossEntropyGradient(backend, 17, 769, 53L, 54L);
    }

    protected static void assertCrossEntropyGradient(MetalBackend backend, int rows, int columns,
                                                   long valueSeed, long targetSeed) {
        Tensor logits = randomTensor(rows, columns, valueSeed);
        int[] targets = randomTargets(rows, columns, targetSeed);
        Tensor expected = cpu.crossEntropyGradient(logits, targets);
        Tensor actual = backend.crossEntropyGradient(logits, targets);
        assertTensorClose(expected, actual, 1e-3f, 1e-2f);
    }

    protected static void assertSingleAdamStep(MetalBackend backend) {
        AdamState expected = adamState(32, 64, 71L);
        AdamState actual = copyState(expected);
        Tensor gradient = randomTensor(32, 64, 72L);
        updateAdam(cpu, expected, gradient, 1);
        updateAdam(backend, actual, new Tensor(gradient), 1);
        assertAdamState(expected, actual);
    }

    protected static void assertMultipleAdamSteps(MetalBackend backend) {
        AdamState expected = adamState(16, 48, 81L);
        AdamState actual = copyState(expected);
        for (int step = 1; step <= 5; step++) {
            Tensor gradient = randomTensor(16, 48, 90L + step);
            updateAdam(cpu, expected, gradient, step);
            updateAdam(backend, actual, new Tensor(gradient), step);
        }
        assertAdamState(expected, actual);
    }

    protected static AdamState adamState(int rows, int columns, long seed) {
        return new AdamState(
                randomTensor(rows, columns, seed), new Tensor(rows, columns),
                new Tensor(rows, columns));
    }

    protected static AdamState copyState(AdamState source) {
        return new AdamState(new Tensor(source.weights()), new Tensor(source.firstMoment()),
                new Tensor(source.secondMoment()));
    }

    protected static void updateAdam(TensorBackend backend, AdamState state,
                                   Tensor gradient, int step) {
        float correction1 = 1.0f - (float) Math.pow(ADAM_BETA1, step);
        float correction2 = 1.0f - (float) Math.pow(ADAM_BETA2, step);
        backend.adamWUpdate(state.weights(), gradient, state.firstMoment(), state.secondMoment(),
                ADAM_LR, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, ADAM_WEIGHT_DECAY,
                correction1, correction2);
    }

    protected static void assertAdamState(AdamState expected, AdamState actual) {
        assertTensorClose(expected.weights(), actual.weights(), 1e-4f, 1e-4f);
        assertTensorClose(expected.firstMoment(), actual.firstMoment(), 1e-4f, 1e-4f);
        assertTensorClose(expected.secondMoment(), actual.secondMoment(), 1e-4f, 1e-4f);
    }

    protected record AdamState(Tensor weights, Tensor firstMoment, Tensor secondMoment) {}

    protected static void assertTensorClose(Tensor expected, Tensor actual, double atol, double rtol) {
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
