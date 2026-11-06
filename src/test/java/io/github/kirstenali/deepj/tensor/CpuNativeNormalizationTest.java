package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class CpuNativeNormalizationTest {
    @BeforeEach void cpu() {
        Tensor.setBackend(new CpuBackend());
        assumeTrue(CpuBlas.AVAILABLE);
    }

    @ParameterizedTest @ValueSource(ints = {1, 7, 64, 257})
    void forwardAndBackwardMatchScalarStridedFallbackBitForBit(int rows) {
        Tensor input = values(rows + 2, 257).slice(0, 1, rows + 1);
        Tensor gamma = values(2, 257).slice(0, 1, 2), beta = values(2, 257).slice(0, 1, 2);
        Tensor gradient = values(rows + 2, 257).slice(0, 1, rows + 1);
        for (boolean centered : new boolean[]{true, false}) {
            Tensor bias = centered ? beta : null;
            var expected = CpuNdNormalization.apply(strided(input), gamma, bias, 1e-5f, centered);
            var actual = CpuNdNormalization.apply(input, gamma, bias, 1e-5f, centered);
            compare(expected, actual);
            Tensor derivative = CpuNdNormalization.backward(gradient, actual.normalized(),
                    actual.denominator(), gamma, centered);
            Tensor reference = CpuNdNormalization.backward(strided(gradient), expected.normalized(),
                    expected.denominator(), gamma, centered);
            assertArrayEquals(reference.toFloatArray(), derivative.toFloatArray());
        }
    }

    private static void compare(NormalizationResult expected, NormalizationResult actual) {
        assertArrayEquals(expected.output().toFloatArray(), actual.output().toFloatArray());
        assertArrayEquals(expected.normalized().toFloatArray(), actual.normalized().toFloatArray());
        assertArrayEquals(expected.denominator().toFloatArray(), actual.denominator().toFloatArray());
    }

    private static Tensor strided(Tensor input) {
        float[] values = input.toFloatArray(), padded = new float[values.length * 2];
        for (int i = 0; i < values.length; i++) { padded[i * 2] = values[i]; }
        return Tensor.from(padded, input.size(0), input.size(1) * 2).slice(1, 0, input.size(1) * 2, 2);
    }

    private static Tensor values(int rows, int width) {
        var random = new Random(37L * rows + width);
        float[] values = new float[rows * width];
        for (int i = 0; i < values.length; i++) { values[i] = random.nextFloat() * 8 - 4; }
        return Tensor.from(values, rows, width);
    }
}
