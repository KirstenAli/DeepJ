package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/** Basic correctness checks for selected Metal backend ops against CPU references. */
public final class MetalBackendTest {

    private static CpuBackend cpu;
    private static TensorBackend gpu;
    private static TensorBackend previousBackend;

    @BeforeAll
    static void setUp() {
        Assumptions.assumeTrue(MetalNative.AVAILABLE, "Metal native library not available");
        cpu = new CpuBackend();
        gpu = new MetalBackend();
        previousBackend = Tensor.backend();
        Tensor.setBackend(gpu); // so materialize() routes through MetalBackend
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
        assertTrue(MetalNative.AVAILABLE);
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
        MetalBackend gpuBackend = new MetalBackend();
        TensorBackend oldBackend = Tensor.backend();
        Tensor.setBackend(gpuBackend);

        try {
            Tensor a = randomTensor(8, 16, 100L);
            Tensor row = randomTensor(1, 16, 101L);
            Tensor col = randomTensor(8, 1, 102L);

            assertTensorClose(cpu.addRowVector(a, row), gpuBackend.addRowVector(a, row), 1e-4f, 1e-4f);
            assertTensorClose(cpu.addBroadcastCols(a, col), gpuBackend.addBroadcastCols(a, col), 1e-4f, 1e-4f);
            assertTensorClose(cpu.subtractBroadcastCols(a, col), gpuBackend.subtractBroadcastCols(a, col), 1e-4f, 1e-4f);
            assertTensorClose(cpu.divideBroadcastCols(a, col), gpuBackend.divideBroadcastCols(a, col), 1e-4f, 1e-4f);
            assertTensorClose(cpu.multiplyBroadcastCols(a, col), gpuBackend.multiplyBroadcastCols(a, col), 1e-4f, 1e-4f);
            assertTensorClose(cpu.multiplyBroadcastRows(a, row), gpuBackend.multiplyBroadcastRows(a, row), 1e-4f, 1e-4f);

            assertTensorClose(cpu.addScalar(a, 0.25f), gpuBackend.addScalar(a, 0.25f), 1e-4f, 1e-4f);
            assertTensorClose(cpu.divideScalar(a, 1.5f), gpuBackend.divideScalar(a, 1.5f), 1e-4f, 1e-4f);
        } finally {
            gpuBackend.releaseResources();
            Tensor.setBackend(oldBackend);
        }
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
        MetalBackend gpuBackend = new MetalBackend();
        TensorBackend oldBackend = Tensor.backend();
        Tensor.setBackend(gpuBackend);

        try {
            Tensor a = randomTensor(16, 20, 121L);
            assertTensorClose(cpu.maxAlongRows(a), gpuBackend.maxAlongRows(a), 1e-4f, 1e-4f);
            assertTensorClose(cpu.clamp(a, -0.25f, 0.35f), gpuBackend.clamp(a, -0.25f, 0.35f), 1e-4f, 1e-4f);
            assertTensorClose(cpu.pow(a, 2.0f), gpuBackend.pow(a, 2.0f), 1e-4f, 1e-4f);

            // Unique indices: exercises GPU scatter path.
            Tensor targetCpuUnique = randomTensor(10, 8, 122L);
            Tensor targetGpuUnique = new Tensor(targetCpuUnique);
            Tensor gradUnique = randomTensor(6, 8, 123L);
            int[] uniqueIndices = new int[]{3, 1, 7, 2, 4, 5};

            cpu.scatterAddRows(targetCpuUnique, uniqueIndices, gradUnique);
            Tensor.scatterAddRows(targetGpuUnique, uniqueIndices, gradUnique);
            assertTensorClose(targetCpuUnique, targetGpuUnique, 1e-4f, 1e-4f);

            // Duplicate indices: uses atomic GPU accumulation.
            Tensor targetCpuDup = randomTensor(10, 8, 124L);
            Tensor targetGpuDup = new Tensor(targetCpuDup);
            Tensor gradDup = randomTensor(6, 8, 125L);
            int[] dupIndices = new int[]{3, 1, 7, 1, 4, 3};

            cpu.scatterAddRows(targetCpuDup, dupIndices, gradDup);
            Tensor.scatterAddRows(targetGpuDup, dupIndices, gradDup);
            assertTensorClose(targetCpuDup, targetGpuDup, 1e-3f, 1e-3f);
        } finally {
            gpuBackend.releaseResources();
            Tensor.setBackend(oldBackend);
        }
    }

    @Test
    void scatterAddRowsDuplicateIndicesAtomicModeMatchesCpu() {
        MetalBackend gpuBackend = new MetalBackend();
        TensorBackend oldBackend = Tensor.backend();
        Tensor.setBackend(gpuBackend);

        try {
            Tensor targetCpuDup = randomTensor(64, 32, 129L);
            Tensor targetGpuDup = new Tensor(targetCpuDup);
            Tensor gradDup = randomTensor(128, 32, 130L);
            int[] dupIndices = new int[128];
            Random rnd = new Random(131L);
            for (int i = 0; i < dupIndices.length; i++) {
                dupIndices[i] = rnd.nextInt(targetCpuDup.rows);
            }

            cpu.scatterAddRows(targetCpuDup, dupIndices, gradDup);
            Tensor.scatterAddRows(targetGpuDup, dupIndices, gradDup);
            assertTensorClose(targetCpuDup, targetGpuDup, 1e-3f, 1e-3f);
        } finally {
            gpuBackend.releaseResources();
            Tensor.setBackend(oldBackend);
        }
    }

    @Test
    void scalarSumAbsAndCrossEntropyLossMatchCpu() {
        MetalBackend gpuBackend = new MetalBackend();
        TensorBackend oldBackend = Tensor.backend();
        Tensor.setBackend(gpuBackend);

        try {
            Tensor a = randomTensor(23, 17, 126L);
            float expectedSumAbs = cpu.sumAbs(a);
            float actualSumAbs = gpuBackend.sumAbs(a);
            assertEquals(expectedSumAbs, actualSumAbs, 1e-4f);

            Tensor logits = randomTensor(19, 31, 127L);
            int[] targets = new int[logits.rows];
            Random rnd = new Random(128L);
            for (int r = 0; r < targets.length; r++) {
                targets[r] = rnd.nextInt(logits.cols);
            }

            float expectedCe = cpu.crossEntropyLoss(logits, targets);
            float actualCe = gpuBackend.crossEntropyLoss(logits, targets);
            assertEquals(expectedCe, actualCe, 1e-4f);

            // Wide-column cases exercise threadgroup-strided column reductions.
            Tensor wideA = randomTensor(29, 513, 132L);
            assertEquals(cpu.sumAbs(wideA), gpuBackend.sumAbs(wideA), 1e-3f);

            Tensor wideLogits = randomTensor(23, 777, 133L);
            int[] wideTargets = new int[wideLogits.rows];
            Random wideRnd = new Random(134L);
            for (int r = 0; r < wideTargets.length; r++) {
                wideTargets[r] = wideRnd.nextInt(wideLogits.cols);
            }
            assertEquals(cpu.crossEntropyLoss(wideLogits, wideTargets),
                    gpuBackend.crossEntropyLoss(wideLogits, wideTargets), 1e-3f);
        } finally {
            gpuBackend.releaseResources();
            Tensor.setBackend(oldBackend);
        }
    }

    @Test
    void softmaxBackwardMatchesCpu() {
        MetalBackend gpuBackend = new MetalBackend();
        TensorBackend oldBackend = Tensor.backend();
        Tensor.setBackend(gpuBackend);

        try {
            Tensor gradOutput = randomTensor(32, 64, 10L);
            Tensor logits = randomTensor(32, 64, 11L);
            Tensor softmaxOut = cpu.softmaxRows(logits);

            Tensor expected = cpu.softmaxBackward(gradOutput, softmaxOut);
            Tensor actual = gpuBackend.softmaxBackward(gradOutput, softmaxOut);

            assertTensorClose(expected, actual, 1e-4f, 1e-4f);

            Tensor gradOutputWide = randomTensor(17, 769, 12L);
            Tensor logitsWide = randomTensor(17, 769, 13L);
            Tensor softmaxOutWide = cpu.softmaxRows(logitsWide);
            Tensor expectedWide = cpu.softmaxBackward(gradOutputWide, softmaxOutWide);
            Tensor actualWide = gpuBackend.softmaxBackward(gradOutputWide, softmaxOutWide);
            assertTensorClose(expectedWide, actualWide, 1e-4f, 1e-4f);
        } finally {
            gpuBackend.releaseResources();
            Tensor.setBackend(oldBackend);
        }
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
        MetalBackend gpuBackend = new MetalBackend();
        TensorBackend oldBackend = Tensor.backend();
        Tensor.setBackend(gpuBackend);

        try {
            Tensor dXHat = randomTensor(16, 32, 31L);
            Tensor xHat = randomTensor(16, 32, 32L);
            Tensor std = new Tensor(16, 1);
            for (int r = 0; r < std.rows; r++) {
                // std is rows×1: flat index r*1+0 = r
                std.data[r] = 0.5f + Math.abs(xHat.data[r * xHat.cols]);
            }

            Tensor expected = cpu.layerNormBackward(dXHat, xHat, std, dXHat.cols);
            Tensor actual = gpuBackend.layerNormBackward(dXHat, xHat, std, dXHat.cols);

            assertTensorClose(expected, actual, 1e-4f, 1e-4f);

            Tensor dXHatWide = randomTensor(11, 777, 33L);
            Tensor xHatWide = randomTensor(11, 777, 34L);
            Tensor stdWide = new Tensor(11, 1);
            for (int r = 0; r < stdWide.rows; r++) {
                stdWide.data[r] = 0.5f + Math.abs(xHatWide.data[r * xHatWide.cols]);
            }
            Tensor expectedWide = cpu.layerNormBackward(dXHatWide, xHatWide, stdWide, dXHatWide.cols);
            Tensor actualWide = gpuBackend.layerNormBackward(dXHatWide, xHatWide, stdWide, dXHatWide.cols);
            assertTensorClose(expectedWide, actualWide, 1e-4f, 1e-4f);
        } finally {
            gpuBackend.releaseResources();
            Tensor.setBackend(oldBackend);
        }
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
        MetalBackend gpuBackend = new MetalBackend();
        TensorBackend oldBackend = Tensor.backend();
        Tensor.setBackend(gpuBackend);

        try {
            Tensor logits = randomTensor(32, 64, 51L);
            int[] targets = new int[logits.rows];
            Random rnd = new Random(52L);
            for (int r = 0; r < targets.length; r++) {
                targets[r] = rnd.nextInt(logits.cols);
            }

            Tensor expected = cpu.crossEntropyGradient(logits, targets);
            Tensor actual = gpuBackend.crossEntropyGradient(logits, targets);

            assertTensorClose(expected, actual, 1e-4f, 1e-4f);

            Tensor wideLogits = randomTensor(17, 769, 53L);
            int[] wideTargets = new int[wideLogits.rows];
            Random wideRnd = new Random(54L);
            for (int r = 0; r < wideTargets.length; r++) {
                wideTargets[r] = wideRnd.nextInt(wideLogits.cols);
            }
            Tensor expectedWide = cpu.crossEntropyGradient(wideLogits, wideTargets);
            Tensor actualWide = gpuBackend.crossEntropyGradient(wideLogits, wideTargets);
            assertTensorClose(expectedWide, actualWide, 1e-4f, 1e-4f);
        } finally {
            gpuBackend.releaseResources();
            Tensor.setBackend(oldBackend);
        }
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
        MetalBackend gpuBackend = new MetalBackend();
        TensorBackend oldBackend = Tensor.backend();
        Tensor.setBackend(gpuBackend);

        try {
            Tensor wCpu = randomTensor(32, 64, 71L);
            Tensor gCpu = randomTensor(32, 64, 72L);
            Tensor mtCpu = new Tensor(32, 64);
            Tensor vtCpu = new Tensor(32, 64);

            Tensor wGpu = new Tensor(wCpu);
            Tensor gGpu = new Tensor(gCpu);
            Tensor mtGpu = new Tensor(32, 64);
            Tensor vtGpu = new Tensor(32, 64);

            float lr = 1e-3f;
            float beta1 = 0.9f;
            float beta2 = 0.999f;
            float eps = 1e-8f;
            float weightDecay = 0.01f;
            float bc1 = 1.0f - beta1;
            float bc2 = 1.0f - beta2;

            cpu.adamWUpdate(wCpu, gCpu, mtCpu, vtCpu, lr, beta1, beta2, eps, weightDecay, bc1, bc2);
            gpuBackend.adamWUpdate(wGpu, gGpu, mtGpu, vtGpu, lr, beta1, beta2, eps, weightDecay, bc1, bc2);

            assertTensorClose(wCpu, wGpu, 1e-4f, 1e-4f);
            assertTensorClose(mtCpu, mtGpu, 1e-4f, 1e-4f);
            assertTensorClose(vtCpu, vtGpu, 1e-4f, 1e-4f);
        } finally {
            gpuBackend.releaseResources();
            Tensor.setBackend(oldBackend);
        }
    }

    @Test
    void adamWUpdateMatchesCpu_multipleSteps() {
        MetalBackend gpuBackend = new MetalBackend();
        TensorBackend oldBackend = Tensor.backend();
        Tensor.setBackend(gpuBackend);

        try {
            Tensor wCpu = randomTensor(16, 48, 81L);
            Tensor mtCpu = new Tensor(16, 48);
            Tensor vtCpu = new Tensor(16, 48);

            Tensor wGpu = new Tensor(wCpu);
            Tensor mtGpu = new Tensor(16, 48);
            Tensor vtGpu = new Tensor(16, 48);

            float lr = 1e-3f;
            float beta1 = 0.9f;
            float beta2 = 0.999f;
            float eps = 1e-8f;
            float weightDecay = 0.01f;

            for (int step = 1; step <= 5; step++) {
                Tensor gCpu = randomTensor(16, 48, 90L + step);
                Tensor gGpu = new Tensor(gCpu);
                float bc1 = 1.0f - (float) Math.pow(beta1, step);
                float bc2 = 1.0f - (float) Math.pow(beta2, step);

                cpu.adamWUpdate(wCpu, gCpu, mtCpu, vtCpu, lr, beta1, beta2, eps, weightDecay, bc1, bc2);
                gpuBackend.adamWUpdate(wGpu, gGpu, mtGpu, vtGpu, lr, beta1, beta2, eps, weightDecay, bc1, bc2);
            }

            assertTensorClose(wCpu, wGpu, 1e-4f, 1e-4f);
            assertTensorClose(mtCpu, mtGpu, 1e-4f, 1e-4f);
            assertTensorClose(vtCpu, vtGpu, 1e-4f, 1e-4f);
        } finally {
            gpuBackend.releaseResources();
            Tensor.setBackend(oldBackend);
        }
    }

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
