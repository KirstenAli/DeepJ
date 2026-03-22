package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Metal backend tests are intentionally limited to matmul.
 *
 * <p>All other {@link TensorBackend} ops are delegated to {@link CpuBackend}.
 */
public final class MetalBackendTest {

    private static TensorBackend cpu;
    private static TensorBackend gpu;

    @BeforeAll
    static void setUp() {
        Assumptions.assumeTrue(MetalNative.AVAILABLE, "Metal native library not available");
        cpu = new CpuBackend();
        gpu = new MetalBackend();
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

    private static void assertTensorClose(Tensor expected, Tensor actual, double atol, double rtol) {
        assertEquals(expected.rows, actual.rows, "rows mismatch");
        assertEquals(expected.cols, actual.cols, "cols mismatch");

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
