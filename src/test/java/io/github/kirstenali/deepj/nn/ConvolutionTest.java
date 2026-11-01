package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import java.util.Random;
import static io.github.kirstenali.deepj.nn.NeuralTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

class ConvolutionTest extends NeuralTest {
    @Test void paddedStridedDilatedConvolutionMatchesIndependentCoordinateSums() {
        var window = new Window2D(2, 2, 2, 1, 1, 1, 2, 1);
        for (DType dtype : DType.values()) {
            Conv2D layer = new Conv2D(2, 3, window, true, new Random(1), new ParameterFactory(dtype));
            initialize(layer);
            Tensor input = values(2, 2, 5, 4).transpose(2, 3).to(dtype);
            compare(layer, input);
        }
    }

    @Test void convolutionAutogradMatchesFiniteDifferencesThroughOverlappingWindows() {
        Conv2D layer = new Conv2D(1, 2, new Window2D(2, 2, 1, 1, 1, 0, 1, 1),
                true, new Random(2), ParameterFactory.float32());
        initialize(layer);
        gradients(layer, values(1, 1, 2, 3));
        Conv2D noBias = new Conv2D(1, 1, new Window2D(1, 2), false,
                new Random(3), ParameterFactory.float32());
        assertNull(noBias.bias());
        assertEquals(1, noBias.parameters().size());
        gradients(noBias, values(1, 1, 2, 3));
    }

    @Test void temporalConvolutionHasTheExpectedKernelOrderAndStateGradients() {
        Conv1D layer = new Conv1D(1, 1, 2, new Random(1));
        try (var ignored = GradMode.noGrad()) {
            layer.weight().value.copyFrom(Tensor.from(new float[]{2, 3}, 2, 1));
            layer.bias().value.copyFrom(Tensor.from(new float[]{1}, 1, 1));
        }
        Tensor input = Tensor.from(new float[]{1, 2, 3, 4}, 1, 1, 4).requiresGrad(true);
        Tensor result = layer.forward(input);
        assertArrayEquals(new float[]{9, 14, 19}, result.toFloatArray());
        result.sumTensor().backward();
        assertArrayEquals(new float[]{2, 5, 5, 3}, input.grad().toFloatArray());
        assertArrayEquals(new float[]{6, 9}, layer.weight().grad().toFloatArray());
        assertEquals(3, layer.bias().grad().item());
    }

    @Test void dilatedTemporalConvolutionComposesIntoATrainableCnn() {
        Conv1D convolution = new Conv1D(2, 3, 2, 2, 1, 2, false,
                new Random(4), ParameterFactory.float32());
        Sequential cnn = new Sequential(convolution, Activation.tanh(), Pool1D.average(2), new GlobalAveragePool());
        initialize(cnn);
        assertNull(convolution.bias());
        gradients(cnn, values(2, 2, 7));
        assertArrayEquals(new int[]{1, 1, 2, 2}, new Conv2D(1, 1, 1, new Random(1))
                .forward(values(1, 1, 2, 2)).shape());
    }

    @Test void paddingDoesNotMultiplyInfinityByZero() {
        Conv2D layer = new Conv2D(1, 1, new Window2D(1, 1, 1, 1, 1, 1, 1, 1), false,
                new Random(2), ParameterFactory.float32());
        try (var ignored = GradMode.noGrad()) { layer.weight().value.copyFrom(Tensor.ones(1, 1)); }
        Tensor output = layer.forward(Tensor.from(new float[]{Float.POSITIVE_INFINITY}, 1, 1, 1, 1));
        assertEquals(0, output.get(0, 0, 0, 0));
        assertEquals(Float.POSITIVE_INFINITY, output.get(0, 0, 1, 1));
    }

    private static void compare(Conv2D layer, Tensor input) {
        Tensor actual = layer.forward(input);
        float[] expected = new float[actual.numel()];
        for (int index = 0; index < expected.length; index++) {
            int x = index % actual.size(3), y = index / actual.size(3) % actual.size(2);
            int channel = index / (actual.size(2) * actual.size(3)) % actual.size(1);
            int sample = index / (actual.size(1) * actual.size(2) * actual.size(3));
            expected[index] = reference(layer, input, sample, channel, y, x);
        }
        assertArrayEquals(expected, actual.toFloatArray(), 0.005f);
    }

    private static float reference(Conv2D layer, Tensor input, int sample, int output, int y, int x) {
        float sum = 0;
        Window2D window = layer.window();
        for (int channel = 0; channel < input.size(1); channel++) {
            for (int ky = 0; ky < window.height(); ky++) {
                sum += row(layer, input, sample, channel, output, y, x, ky);
            }
        }
        Tensor rounded = Tensor.from(new float[]{sum}, layer.weight().value.dtype());
        return rounded.item() + layer.bias().value.get(0, output);
    }

    private static float row(Conv2D layer, Tensor input, int sample, int channel, int output, int y, int x, int ky) {
        Window2D window = layer.window();
        int iy = y * window.strideHeight() - window.paddingHeight() + ky * window.dilationHeight();
        float sum = 0;
        if (iy < 0 || iy >= input.size(2)) return sum;
        for (int kx = 0; kx < window.width(); kx++) {
            int ix = x * window.strideWidth() - window.paddingWidth() + kx * window.dilationWidth();
            if (ix >= 0 && ix < input.size(3)) {
                int weight = channel * window.area() + ky * window.width() + kx;
                sum += input.get(sample, channel, iy, ix) * layer.weight().value.get(weight, output);
            }
        }
        return sum;
    }
}
