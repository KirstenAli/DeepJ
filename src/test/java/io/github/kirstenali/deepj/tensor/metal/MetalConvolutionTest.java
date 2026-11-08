package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.nn.DepthwiseConv2D;
import io.github.kirstenali.deepj.nn.GroupedConv2D;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorTestStorage;
import java.util.Random;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class MetalConvolutionTest extends MetalDifferentialTestSupport {
    @ParameterizedTest @EnumSource(DType.class)
    void groupedNonSquareKernelsMatchCoordinateSumsAtEveryPaddedEdge(DType dtype) {
        compare(values(dtype, 1, 2, 4, 5, 7), values(dtype, 2, 6, 2, 3, 5), 2);
    }

    @ParameterizedTest @EnumSource(DType.class)
    void depthwiseAndPointwiseKernelsKeepIndependentChannels(DType dtype) {
        compare(values(dtype, 3, 1, 4, 3, 5), values(dtype, 4, 4, 1, 3, 3), 4);
        compare(values(dtype, 5, 2, 4, 3, 5), values(dtype, 6, 6, 2, 1, 1), 2);
    }

    @ParameterizedTest @EnumSource(DType.class)
    void temporalKernelSlicesAndStridedOffsetInputsStayOnDevice(DType dtype) {
        Tensor input = values(dtype, 7, 3, 4, 9, 7).slice(0, 1, 3).slice(2, 1, 8, 2).permute(0, 1, 3, 2);
        Tensor kernel = values(dtype, 8, 6, 2, 3, 3, 5).select(2, 1);
        compare(input, kernel, 2);
    }

    private void compare(Tensor input, Tensor kernel, int groups) {
        Tensor expected = reference(input, kernel, groups);
        try (var ignored = GradMode.noGrad()) {
            Tensor result = GroupedConv2D.apply(input, kernel, null, groups);
            assertFalse(TensorTestStorage.hasHostStorage(result));
            assertEquals(input.dtype(), result.dtype());
            float tolerance = input.dtype() == DType.FLOAT32 ? 2e-5f : 0.03f;
            assertTensorClose(expected, result, tolerance, tolerance);
        }
    }

    private Tensor reference(Tensor input, Tensor weight, int groups) {
        int[] shape = {input.size(0), weight.size(0), input.size(2), input.size(3)};
        float[] source = input.toFloatArray(), kernel = weight.toFloatArray();
        float[] output = new float[shape[0] * shape[1] * shape[2] * shape[3]];
        for (int index = 0; index < output.length; index++) {
            output[index] = coordinate(source, kernel, input.shape(), weight.shape(), groups, index);
        }
        return Tensor.from(output, input.dtype(), shape);
    }

    private static float coordinate(float[] input, float[] weight, int[] s, int[] k, int groups, int index) {
        int x = index % s[3], y = index / s[3] % s[2];
        int channel = index / (s[2] * s[3]) % k[0], batch = index / (s[2] * s[3] * k[0]);
        int first = channel / (k[0] / groups) * k[1];
        float sum = 0;
        for (int c = 0; c < k[1]; c++) {
            sum += plane(input, weight, s, k, batch, first + c, channel * k[1] + c, y, x);
        }
        return sum;
    }

    private static float plane(float[] input, float[] weight, int[] s, int[] k,
                               int batch, int channel, int kernel, int y, int x) {
        float sum = 0;
        for (int row = 0; row < k[2]; row++) {
            for (int col = 0; col < k[3]; col++) {
                int iy = y + row - k[2] / 2, ix = x + col - k[3] / 2;
                if (iy >= 0 && iy < s[2] && ix >= 0 && ix < s[3]) {
                    sum += input[((batch * s[1] + channel) * s[2] + iy) * s[3] + ix]
                            * weight[(kernel * k[2] + row) * k[3] + col];
                }
            }
        }
        return sum;
    }

    private static Tensor values(DType dtype, long seed, int... shape) {
        int count = java.util.Arrays.stream(shape).reduce(1, Math::multiplyExact);
        float[] values = new float[count];
        Random random = new Random(seed);
        for (int index = 0; index < count; index++) { values[index] = random.nextFloat() - 0.5f; }
        return Tensor.from(values, dtype, shape);
    }

    @Test void trainingRetainsInputWeightAndBiasGradients() {
        Tensor input = Tensor.from(new float[]{1, 2, 3, 4}, 1, 2, 1, 2).requiresGrad(true);
        Tensor weight = Tensor.from(new float[]{2, 3}, 2, 1, 1, 1).requiresGrad(true);
        Tensor bias = Tensor.from(new float[]{1, -1}, 2).requiresGrad(true);
        DepthwiseConv2D.apply(input, weight, bias).sumTensor().backward();
        assertArrayEquals(new float[]{2, 2, 3, 3}, input.grad().toFloatArray());
        assertArrayEquals(new float[]{3, 7}, weight.grad().toFloatArray());
        assertArrayEquals(new float[]{2, 2}, bias.grad().toFloatArray());
    }

    @Test void unsupportedShapesAndMixedPrecisionAreRejected() {
        Tensor input = Tensor.ones(DType.FLOAT32, 1, 4, 5, 7);
        Tensor weight = Tensor.ones(DType.FLOAT32, 6, 2, 3, 3);
        assertFalse(metal.supportsConvolution2D(input, weight, 0));
        assertFalse(metal.supportsConvolution2D(input, weight.to(DType.FLOAT16), 2));
        assertFalse(metal.supportsConvolution2D(input.reshape(4, 5, 7), weight, 2));
        assertThrows(IllegalArgumentException.class, () -> metal.convolution2D(input, weight, 3));
    }
}
