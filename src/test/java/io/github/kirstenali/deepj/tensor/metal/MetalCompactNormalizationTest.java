package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.Random;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

class MetalCompactNormalizationTest extends MetalDifferentialTestSupport {
    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void parallelStatisticsAndAdjointsMatchDecodedCpuArithmetic(DType type) {
        for (int width : new int[]{127, 128, 129, 2048}) {
            for (boolean centered : new boolean[]{false, true}) {
                compare(type, type, width, centered);
                compare(DType.FLOAT32, type, width, centered);
            }
        }
    }

    private void compare(DType inputType, DType parameterType, int width, boolean centered) {
        Tensor input = values(inputType, 3, width * 2).slice(1, 0, width * 2, 2);
        Tensor gain = values(parameterType, 1, width), bias = values(parameterType, 1, width);
        var expected = cpu.normalize(input, gain, centered ? bias : null, 1e-5f, centered);
        var actual = metal.normalize(input, gain, centered ? bias : null, 1e-5f, centered);
        float tolerance = actual.output().dtype() == DType.BFLOAT16 ? .008f : .001f;
        assertTensorClose(expected.output(), actual.output(), 2e-5f, tolerance);
        assertTensorClose(expected.normalized(), actual.normalized(), 3e-6f, 3e-6f);
        assertTensorClose(expected.denominator(), actual.denominator(), 3e-6f, 3e-6f);
        Tensor upstream = values(DType.FLOAT32, 3, width);
        Tensor reference = cpu.normalizationBackward(upstream, expected.normalized(),
                expected.denominator(), gain, centered);
        Tensor gradient = metal.normalizationBackward(upstream, actual.normalized(),
                actual.denominator(), gain, centered);
        assertTensorClose(reference, gradient, 3e-6f, 5e-6f);
    }

    private static Tensor values(DType dtype, int rows, int width) {
        float[] data = new float[rows * width];
        var random = new Random(281 + rows + width);
        for (int index = 0; index < data.length; index++) { data[index] = random.nextFloat() - .5f; }
        return Tensor.from(data, dtype, rows, width);
    }
}
