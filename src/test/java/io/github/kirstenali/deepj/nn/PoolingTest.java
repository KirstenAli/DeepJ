package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import static io.github.kirstenali.deepj.nn.NeuralTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

class PoolingTest extends NeuralTest {
    @Test void maxAndAveragePoolMatchKnownSpatialValues() {
        Tensor input = Tensor.from(new float[]{1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16}, 1, 1, 4, 4);
        assertArrayEquals(new float[]{4, 8, 12, 16}, Pool2D.max(2).forward(input).toFloatArray());
        assertArrayEquals(new float[]{2.5f, 6.5f, 10.5f, 14.5f}, Pool2D.average(2).forward(input).toFloatArray());
        assertArrayEquals(new float[]{8.5f}, new GlobalAveragePool().forward(input).toFloatArray());
    }

    @Test void maxPoolPaddingIsNegativeInfinityAndTiesShareGradients() {
        Tensor input = Tensor.from(new float[]{-2, -2}, 1, 1, 2).requiresGrad(true);
        Pool1D pooling = new Pool1D(2, 1, 1, Pool2D.Reduction.MAX);
        Tensor output = pooling.forward(input);
        assertArrayEquals(new float[]{-2, -2, -2}, output.toFloatArray());
        output.sumTensor().backward();
        assertArrayEquals(new float[]{1.5f, 1.5f}, input.grad().toFloatArray());
        assertEquals(-2, Pool1D.max(2).forward(input).item());
    }

    @Test void averagePoolingIncludesPaddedZerosAndOverlappingGradientsAccumulate() {
        Tensor input = Tensor.from(new float[]{2, 4}, 1, 1, 2).requiresGrad(true);
        Tensor output = new Pool1D(2, 1, 1, Pool2D.Reduction.AVERAGE).forward(input);
        assertArrayEquals(new float[]{1, 3, 2}, output.toFloatArray());
        output.sumTensor().backward();
        assertArrayEquals(new float[]{1, 1}, input.grad().toFloatArray());
        gradients(Pool2D.max(2), values(1, 1, 4, 4));
        gradients(Pool2D.average(2), values(2, 2, 2, 2));
        gradients(new GlobalAveragePool(), values(2, 3, 2, 2));
    }

    @Test void invalidSpatialConfigurationsFailBeforeAllocatingWindows() {
        assertThrows(IllegalArgumentException.class, () -> new Window2D(0, 1));
        assertThrows(IllegalArgumentException.class, () -> new Window2D(1, 0));
        assertThrows(IllegalArgumentException.class, () -> new Window2D(1, 1, 0, 1, 0, 0, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> new Window2D(1, 1, 1, 0, 0, 0, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> new Window2D(1, 1, 1, 1, -1, 0, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> new Window2D(1, 1, 1, 1, 0, -1, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> new Window2D(1, 1, 1, 1, 0, 0, 0, 1));
        assertThrows(IllegalArgumentException.class, () -> new Window2D(1, 1, 1, 1, 0, 0, 1, 0));
        assertThrows(IllegalArgumentException.class, () -> Pool2D.max(4).forward(values(1, 1, 2, 2)));
        assertThrows(IllegalArgumentException.class, () -> Pool1D.max(2).forward(values(2, 2)));
        assertThrows(IllegalArgumentException.class, () -> new GlobalAveragePool().forward(values(2, 2)));
    }

    @Test void convolutionRejectsInvalidChannelsAndRanks() {
        assertThrows(IllegalArgumentException.class, () -> new Conv2D(0, 1, 1, new java.util.Random(1)));
        assertThrows(IllegalArgumentException.class, () -> new Conv2D(1, 0, 1, new java.util.Random(1)));
        Conv2D layer = new Conv2D(2, 1, 1, new java.util.Random(1));
        assertThrows(IllegalArgumentException.class, () -> layer.forward(values(1, 1, 2, 2)));
        assertThrows(IllegalArgumentException.class, () -> layer.forward(values(2, 2)));
        assertThrows(IllegalArgumentException.class, () -> layer.forward(new Tensor(0, 2, 2, 2)));
        assertThrows(IllegalArgumentException.class, () -> layer.forward(null));
        assertThrows(IllegalArgumentException.class, () -> new Window2D(1, 1).outputHeight(0));
    }
}
