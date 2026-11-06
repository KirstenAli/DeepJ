package io.github.kirstenali.deepj.tensor.cpu;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

class CpuCausalSoftmaxTest {
    private final CpuBackend cpu = new CpuBackend();
    @BeforeEach void backend() { Tensor.setBackend(cpu); }

    @Test
    void fusedRowsPreserveTheExistingThreePassResultBitsAndInputs() {
        boolean previous = DeepJExecutor.isParallelEnabled();
        try {
            for (boolean parallel : new boolean[]{false, true}) {
                DeepJExecutor.setParallelEnabled(parallel);
                checkWidths();
            }
        } finally {
            DeepJExecutor.setParallelEnabled(previous);
        }
    }

    private void checkWidths() {
        for (int width : new int[]{1, 4, 64, 257}) {
            float[] values = new float[3 * width * width];
            var random = new Random(width);
            for (int i = 0; i < values.length; i++) { values[i] = random.nextFloat() * 16 - 8; }
            checkScales(Tensor.from(values, 3 * width, width));
        }
    }

    @Test
    void maskedNonfiniteExtremeAndEmptyValuesRetainLegacyBehavior() {
        checkScales(Tensor.from(new float[]{-1e12f, Float.NaN, -1e10f, -1e11f}, 2, 2));
        checkScales(Tensor.from(new float[]{Float.NaN, 2, 0, 1}, 2, 2));
        checkScales(Tensor.from(new float[]{Float.POSITIVE_INFINITY, 0, Float.NEGATIVE_INFINITY, 1}, 2, 2));
        checkScales(new Tensor(0, 4));
        assertThrows(IllegalArgumentException.class, () -> cpu.causalSoftmax(new Tensor(3, 4), 4, 1));
        assertThrows(IllegalArgumentException.class, () -> cpu.causalSoftmax(new Tensor(4, 4), 3, 1));
    }

    private void checkScales(Tensor input) {
        float[] original = input.toFloatArray();
        for (float scale : new float[]{0.125f, -0.5f, -0.0f, Float.POSITIVE_INFINITY, Float.NaN}) {
            int width = input.matrixCols();
            Tensor expected = cpu.softmaxRows(cpu.causalMask(cpu.multiplyScalar(input, scale), width));
            Tensor actual = cpu.causalSoftmax(input, width, scale);
            assertBits(expected.toFloatArray(), actual.toFloatArray());
            assertArrayEquals(input.shape(), actual.shape());
        }
        assertBits(original, input.toFloatArray());
        assertEquals(0, input.gradientState().version());
    }

    private static void assertBits(float[] expected, float[] actual) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(Float.floatToIntBits(expected[i]), Float.floatToIntBits(actual[i]));
        }
    }
}
