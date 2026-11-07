package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class CpuCompactNormalizationTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void compactSavedStatisticsRetainTheGenericBackwardContract(DType type) {
        Tensor gradient = values(type, 3, 257), gamma = values(type, 1, 257);
        Tensor normalized = values(type, 3, 257), denominator = Tensor.ones(type, 3, 1);
        for (boolean centered : new boolean[]{false, true}) {
            Tensor expected = CpuNdNormalization.backward(strided(gradient), normalized, denominator, gamma, centered);
            Tensor actual = CpuNdNormalization.backward(gradient, normalized, denominator, gamma, centered);
            assertArrayEquals(expected.toFloatArray(), actual.toFloatArray());
        }
    }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void compactAndMixedStatisticsAndGradientsMatchScalarOrder(DType type) {
        for (int width : new int[]{127, 257, 2048, 8192, 8193}) {
            for (boolean centered : new boolean[]{true, false}) {
                compare(type, type, width, centered);
                compare(type, DType.FLOAT32, width, centered);
                compare(DType.FLOAT32, type, width, centered);
            }
        }
    }

    private static void compare(DType inputType, DType weightType, int width, boolean centered) {
        Tensor input = values(inputType, 5, width).slice(0, 1, 4);
        Tensor gamma = values(weightType, 2, width).slice(0, 1, 2);
        Tensor beta = centered ? gamma.multiplyScalar(.5f) : null;
        var expected = CpuNdNormalization.apply(strided(input), gamma, beta, 1e-5f, centered);
        var actual = CpuNdNormalization.apply(input, gamma, beta, 1e-5f, centered);
        assertArrayEquals(expected.output().toFloatArray(), actual.output().toFloatArray());
        assertArrayEquals(expected.normalized().toFloatArray(), actual.normalized().toFloatArray());
        assertArrayEquals(expected.denominator().toFloatArray(), actual.denominator().toFloatArray());
        for (DType type : new DType[]{inputType, DType.FLOAT32}) {
            Tensor gradient = values(type, 5, width).slice(0, 1, 4);
            Tensor reference = CpuNdNormalization.backward(strided(gradient), expected.normalized(),
                    expected.denominator(), gamma, centered);
            Tensor result = CpuNdNormalization.backward(gradient, actual.normalized(),
                    actual.denominator(), gamma, centered);
            assertArrayEquals(reference.toFloatArray(), result.toFloatArray());
        }
    }

    private static Tensor strided(Tensor input) {
        float[] values = input.toFloatArray(), padded = new float[values.length * 2];
        for (int index = 0; index < values.length; index++) { padded[index * 2] = values[index]; }
        return Tensor.from(padded, input.dtype(), input.size(0), input.size(1) * 2)
                .slice(1, 0, input.size(1) * 2, 2);
    }

    private static Tensor values(DType type, int rows, int width) {
        var random = new Random(81L * rows + width);
        float[] values = new float[rows * width];
        for (int index = 0; index < values.length; index++) { values[index] = random.nextFloat() * 8 - 4; }
        return Tensor.from(values, type, rows, width);
    }
}
