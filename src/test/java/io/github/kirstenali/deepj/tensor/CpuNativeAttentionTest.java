package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class CpuNativeAttentionTest {
    @BeforeEach void cpu() {
        Tensor.setBackend(new CpuBackend());
        assumeTrue(CpuBlas.AVAILABLE);
    }

    @ParameterizedTest @ValueSource(ints = {32, 64, 128, 257, 513})
    void vectorExponentialsPreserveMaskAndJavaRounding(int width) {
        var random = new Random(width);
        float[] values = new float[(width <= 64 ? 4 : 1) * width * width];
        for (int i = 0; i < values.length; i++) { values[i] = random.nextFloat() * 40 - 20; }
        compare(values, width, .125f);
        values[0] = -1e10f;
        compare(values, width, 1f);
        values[width + 1] = Float.POSITIVE_INFINITY;
        compare(values, width, 1f);
        values[2 * width] = Float.NaN;
        compare(values, width, 1f);
    }

    @ParameterizedTest @ValueSource(booleans = {false, true})
    void parallelRowsKeepScalarOrder(boolean parallel) {
        boolean previous = DeepJExecutor.isParallelEnabled();
        try {
            DeepJExecutor.setParallelEnabled(parallel);
            float[] values = Tensor.random(1024, 64, new Random(88)).toFloatArray();
            compare(values, 64, .125f);
        } finally { DeepJExecutor.setParallelEnabled(previous); }
    }

    @ParameterizedTest @ValueSource(ints = {1, 4})
    void fallbackRetainsNativeRowsAndGlobalCausalPositions(int workers) {
        float[] values = new float[128 * 64], output = new float[values.length];
        values[3 * 64] = Float.NaN;
        values[70 * 64] = Float.POSITIVE_INFINITY;
        int[] fallback = new int[128], expected = new int[128];
        expected[3] = expected[70] = 1;
        assertTrue(CpuBlas.causalSoftmax(values, output, 128, 64, .125f, workers, fallback));
        assertArrayEquals(expected, fallback);
        compare(values, 64, .125f);
    }

    @ParameterizedTest @ValueSource(booleans = {false, true})
    void callerWithoutRowFallbackRetainsItsAcceptanceContract(boolean sensitive) {
        float[] values = new float[64 * 64];
        if (sensitive) values[0] = Float.NaN;
        Tensor input = Tensor.from(values, 64, 64), output = new Tensor(64, 64);
        boolean accepted = CpuNativeAttention.softmax(input, output, 64, .125f);
        if (sensitive) assertFalse(accepted);
        else assertTrue(accepted);
    }

    private static void compare(float[] values, int width, float scale) {
        float[] expected = values.clone();
        for (int row = 0; row < values.length / width; row++) { row(expected, row, width, scale); }
        Tensor input = Tensor.from(values, values.length / width, width);
        assertArrayEquals(expected, Tensor.backend().causalSoftmax(input, width, scale).toFloatArray());
        assertArrayEquals(values, input.toFloatArray());
    }

    private static void row(float[] data, int row, int width, float scale) {
        int base = row * width;
        float maximum = Float.NEGATIVE_INFINITY, total = 0;
        for (int i = 0; i < width; i++) {
            data[base + i] = i <= row % width ? data[base + i] * scale : -1e9f;
            if (data[base + i] > maximum) maximum = data[base + i];
        }
        for (int i = 0; i < width; i++) {
            data[base + i] = (float) Math.exp(data[base + i] - maximum);
            total += data[base + i];
        }
        for (int i = 0; i < width; i++) { data[base + i] /= total; }
    }
}
