package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class DepthwiseConv2DTest {
    @BeforeEach void backend() {
        Tensor.setBackend(Boolean.getBoolean("sanavideo.test.metal") ? new MetalBackend() : new CpuBackend());
    }

    @Test void channelTilingPreservesIndependentKernelsAndSpatialEdges() {
        try (var ignored = GradMode.noGrad()) {
            Tensor input = Tensor.ones(DType.FLOAT32, 1, 100, 128, 128);
            float[] scales = new float[100];
            for (int i = 0; i < scales.length; i++) { scales[i] = i + 1; }
            Tensor kernel = Tensor.ones(DType.FLOAT32, 100, 1, 3, 3)
                    .multiply(Tensor.from(scales, 100, 1, 1, 1));
            float[] output = DepthwiseConv2D.apply(input, kernel, null).toFloatArray();
            for (int channel : new int[]{0, 55, 56, 57, 99}) {
                int offset = channel * 128 * 128;
                assertEquals(4 * (channel + 1), output[offset]);
                assertEquals(6 * (channel + 1), output[offset + 64]);
                assertEquals(9 * (channel + 1), output[offset + 64 * 128 + 64]);
            }
        }
    }

    @Test void eachChannelHasItsOwnBiasAndGradient() {
        Tensor input = Tensor.from(new float[]{1, 2, 3, 4}, 1, 2, 1, 2).requiresGrad(true);
        Tensor weight = Tensor.from(new float[]{2, 3}, 2, 1, 1, 1).requiresGrad(true);
        Tensor bias = Tensor.from(new float[]{1, -1}, 2).requiresGrad(true);
        Tensor output = DepthwiseConv2D.apply(input, weight, bias);
        assertArrayEquals(new float[]{3, 5, 8, 11}, output.toFloatArray());
        output.sumTensor().backward();
        assertArrayEquals(new float[]{2, 2, 3, 3}, input.grad().toFloatArray());
        assertArrayEquals(new float[]{3, 7}, weight.grad().toFloatArray());
        assertArrayEquals(new float[]{2, 2}, bias.grad().toFloatArray());
    }
}
