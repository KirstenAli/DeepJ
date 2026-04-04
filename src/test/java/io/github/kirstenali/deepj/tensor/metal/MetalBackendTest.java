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

    private static TensorBackend cpu;
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
        Tensor a = new Tensor(2, 3);
        a.data[0][0] = 1;
        a.data[0][1] = 2;
        a.data[0][2] = 3;
        a.data[1][0] = 4;
        a.data[1][1] = 5;
        a.data[1][2] = 6;

        Tensor b = new Tensor(3, 2);
        b.data[0][0] = 7;
        b.data[0][1] = 8;
        b.data[1][0] = 9;
        b.data[1][1] = 10;
        b.data[2][0] = 11;
        b.data[2][1] = 12;

        Tensor c = gpu.matmul(a, b);
        c.materialize(); // lazy GPU op — must materialize before reading data
        assertEquals(2, c.rows);
        assertEquals(2, c.cols);
        assertEquals(58.0, c.data[0][0], 1e-6);
        assertEquals(64.0, c.data[0][1], 1e-6);
        assertEquals(139.0, c.data[1][0], 1e-6);
        assertEquals(154.0, c.data[1][1], 1e-6);
    }

    @Test
    void matmulMatchesCpu_rectangular() {
        Tensor a = randomTensor(128, 192, 1L);
        Tensor b = randomTensor(192, 64, 2L);

        Tensor expected = cpu.matmul(a, b);
        Tensor actual = gpu.matmul(a, b);
        assertTensorClose(expected, actual, 1e-4, 1e-4);
    }

    @Test
    void matmulRejectsShapeMismatch() {
        Tensor a = randomTensor(3, 4, 3L);
        Tensor b = randomTensor(5, 2, 4L);

        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> gpu.matmul(a, b));
        assertTrue(ex.getMessage().contains("Shape mismatch"));
    }

    @Test
    void softmaxBackwardMatchesCpu() {
        MetalBackend gpuBackend = new MetalBackend(1L, 1L, false);
        TensorBackend oldBackend = Tensor.backend();
        Tensor.setBackend(gpuBackend);

        try {
            Tensor gradOutput = randomTensor(32, 64, 10L);
            Tensor logits = randomTensor(32, 64, 11L);
            Tensor softmaxOut = cpu.softmaxRows(logits);

            Tensor expected = cpu.softmaxBackward(gradOutput, softmaxOut);
            Tensor actual = gpuBackend.softmaxBackward(gradOutput, softmaxOut);

            assertTensorClose(expected, actual, 1e-4, 1e-4);
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
        MetalBackend gpuBackend = new MetalBackend(1L, 1L, false);
        TensorBackend oldBackend = Tensor.backend();
        Tensor.setBackend(gpuBackend);

        try {
            Tensor dXHat = randomTensor(16, 32, 31L);
            Tensor xHat = randomTensor(16, 32, 32L);
            Tensor std = new Tensor(16, 1);
            for (int r = 0; r < std.rows; r++) {
                std.data[r][0] = 0.5 + Math.abs(xHat.data[r][0]);
            }

            Tensor expected = cpu.layerNormBackward(dXHat, xHat, std, dXHat.cols);
            Tensor actual = gpuBackend.layerNormBackward(dXHat, xHat, std, dXHat.cols);

            assertTensorClose(expected, actual, 1e-4, 1e-4);
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
        assertTrue(ex.getMessage().contains("layerNormBackward std"));
    }

    @Test
    void crossEntropyGradientMatchesCpu() {
        MetalBackend gpuBackend = new MetalBackend(1L, 1L, false);
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

            assertTensorClose(expected, actual, 1e-4, 1e-4);
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

    private static void assertTensorClose(Tensor expected, Tensor actual, double atol, double rtol) {
        assertEquals(expected.rows, actual.rows, "rows mismatch");
        assertEquals(expected.cols, actual.cols, "cols mismatch");
        expected.materialize();
        actual.materialize();

        for (int r = 0; r < expected.rows; r++) {
            for (int c = 0; c < expected.cols; c++) {
                double e = expected.data[r][c];
                double a = actual.data[r][c];
                double tol = atol + rtol * Math.abs(e);
                assertTrue(Math.abs(e - a) <= tol, "Mismatch at (" + r + "," + c + ") expected=" + e + " actual=" + a);
            }
        }
    }
}
