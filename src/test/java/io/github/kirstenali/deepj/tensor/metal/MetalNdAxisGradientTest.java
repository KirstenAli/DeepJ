package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertTensorGradients;
import static io.github.kirstenali.deepj.testing.NdStatisticsGradientChecks.assertMeanAndVariance;
import static org.junit.jupiter.api.Assertions.*;

class MetalNdAxisGradientTest extends MetalDifferentialTestSupport {
    @Test
    void meanVarianceAndBroadcastReductionsMatchFiniteDifferences() {
        Tensor input = Tensor.from(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, 2, 1, 3);
        assertMeanAndVariance();
        Tensor scalar = Tensor.from(new float[]{0.2f}, new int[0]);
        assertTensorGradients(scalar, value -> value.add(input).sumAxes(0, 2), Tensor.from(new float[]{0.3f}, 1),
                1e-3f, 1e-4f, 1e-3f);
    }

    @Test
    void softmaxAndSliceSelectAdjointsMatchFiniteDifferences() {
        Tensor input = Tensor.from(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, 2, 1, 3);
        Tensor upstream = Tensor.from(new float[]{0.2f, 0.3f, 0.1f, 0.4f, -0.1f, 0.5f}, 3, 2, 1);
        assertTensorGradients(input, value -> value.permute(2, 0, 1).softmax(1), upstream,
                1e-3f, 1e-4f, 1e-3f);
        assertTensorGradients(input, value -> value.slice(2, 0, 3, 2).select(0, 1),
                Tensor.from(new float[]{0.3f, 0.7f}, 1, 2), 1e-3f, 1e-4f, 1e-3f);
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void maximaCountTiesAndRestoreTheOriginalStridedInputCoordinates(DType dtype) {
        Tensor source = Tensor.from(new float[]{3, 3, 1, 2, 1, 2, 4, 4}, dtype, 2, 2, 2).requiresGrad(true);
        source.permute(2, 0, 1).maximum(1, false).sumTensor().backward();
        Tensor expected = Tensor.from(new float[]{1, 1, 0, 0, 0, 0, 1, 1}, 2, 2, 2);
        assertArrayEquals(expected.toFloatArray(), source.grad().toFloatArray());
        assertEquals(DType.FLOAT32, source.grad().dtype());
        Tensor ties = Tensor.from(new float[]{3, 3, 1, 2}, dtype, 2, 2).requiresGrad(true);
        ties.maximum(1, false).sumTensor().backward();
        assertArrayEquals(new float[]{0.5f, 0.5f, 0, 1}, ties.grad().toFloatArray());
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void varianceUsesFloat32CentersAndSoftmaxGradientsUseStoredProbabilities(DType dtype) {
        Tensor source = Tensor.from(new float[]{0.7f, 1.3f, 2.1f}, dtype, 1, 1, 3).requiresGrad(true);
        float[] values = source.toFloatArray();
        float mean = (values[0] + values[1] + values[2]) / 3;
        source.variance(2, false).backward();
        assertArrayEquals(new float[]{(values[0] - mean) * (2.0f / 3),
                (values[1] - mean) * (2.0f / 3), (values[2] - mean) * (2.0f / 3)},
                source.grad().toFloatArray(), 3e-6f);
        source.zeroGrad();
        Tensor output = source.softmax(2), upstream = Tensor.from(new float[]{0.2f, -0.3f, 0.7f}, 1, 1, 3);
        Tensor expected = cpu.softmaxAxisBackward(upstream, new Tensor(output), 2);
        output.backward(upstream);
        assertArrayEquals(expected.toFloatArray(), source.grad().toFloatArray(), 3e-6f);
        assertEquals(DType.FLOAT32, source.grad().dtype());
    }

    @Test
    void undefinedStatisticsAndNanMaximaKeepExplicitGradientValues() {
        Tensor singleton = Tensor.from(new float[]{1}, 1, 1, 1).requiresGrad(true);
        singleton.variance(2, false, 1).backward();
        assertTrue(Float.isNaN(singleton.grad().item()));
        Tensor source = Tensor.from(new float[]{1, Float.NaN, 2}, 1, 1, 3).requiresGrad(true);
        source.maxAxes().backward();
        for (float gradient : source.grad().toFloatArray()) { assertTrue(Float.isNaN(gradient)); }
        Tensor empty = new Tensor(0, 3, 2).requiresGrad(true);
        empty.maximum(2, false).sumTensor().backward();
        assertArrayEquals(new int[]{0, 3, 2}, empty.grad().shape());
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void slicedCompactInputsAndBroadcastOperandsAccumulateFloat32Gradients(DType dtype) {
        Tensor source = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, dtype, 1, 2, 3).requiresGrad(true);
        Tensor bias = Tensor.from(new float[]{1, 2, 3}, dtype).requiresGrad(true);
        source.add(bias).slice(2, 0, 3, 2).sumAxes().backward();
        assertArrayEquals(new float[]{1, 0, 1, 1, 0, 1}, source.grad().toFloatArray());
        assertArrayEquals(new float[]{2, 0, 2}, bias.grad().toFloatArray());
        assertEquals(DType.FLOAT32, source.grad().dtype());
        assertEquals(DType.FLOAT32, bias.grad().dtype());
    }
}
