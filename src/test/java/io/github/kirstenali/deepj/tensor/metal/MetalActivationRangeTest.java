package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class MetalActivationRangeTest extends MetalDifferentialTestSupport {
    @ParameterizedTest @EnumSource(DType.class)
    void geluLargeFiniteValuesAndGradientsMatchReference(DType dtype) {
        check(dtype, false, true);
        check(dtype, true, true);
    }

    @ParameterizedTest @EnumSource(DType.class)
    void tanhLargeFiniteValuesAndGradientsMatchReference(DType dtype) {
        check(dtype, false, false);
        check(dtype, true, false);
    }

    private static void check(DType dtype, boolean strided, boolean gelu) {
        Tensor input = input(dtype, strided).requiresGrad(true);
        float[][] reference = reference(input.toFloatArray(), gelu);
        Tensor output = gelu ? input.geluActivation() : input.tanhActivation();
        float tolerance = dtype == DType.FLOAT32 ? 2e-5f : 0.008f;
        assertArrayEquals(reference[0], output.toFloatArray(), tolerance);
        output.sumTensor().backward();
        assertArrayEquals(reference[1], input.grad().toFloatArray(), tolerance);
    }

    private static float[][] reference(float[] values, boolean gelu) {
        float[] expected = new float[values.length], gradients = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            double x = values[i], t = gelu ? Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x) : x;
            double h = Math.tanh(t), sech = 1 - h * h;
            expected[i] = (float) (gelu ? 0.5 * x * (1 + h) : h);
            gradients[i] = (float) (gelu ? 0.5 * (1 + h) + 0.5 * x * sech
                    * Math.sqrt(2 / Math.PI) * (1 + 3 * 0.044715 * x * x) : sech);
        }
        return new float[][]{expected, gradients};
    }

    private static Tensor input(DType dtype, boolean strided) {
        float[] values = {-1000, -200, -20, -8, -3, -1, 0, 1, 3, 8, 20, 200, 1000};
        if (!strided) return Tensor.from(values, dtype, 1, values.length);
        float[] paired = new float[values.length * 2];
        for (int i = 0; i < paired.length; i++) { paired[i] = values[i / 2]; }
        return Tensor.from(paired, dtype, values.length, 2).transpose();
    }
}
