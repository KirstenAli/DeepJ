package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class CpuNativeAttentionTest {
    @BeforeEach void cpu() {
        Tensor.setBackend(new CpuBackend());
        assumeTrue(CpuBlas.AVAILABLE);
    }

    @ParameterizedTest @ValueSource(ints = {32, 64, 257, 513})
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
