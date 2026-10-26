package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertTensorGradients;
import static org.junit.jupiter.api.Assertions.*;

class NdStatisticsGradientTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void meanAndCorrectedVarianceMatchFiniteDifferencesAcrossSeveralAxes() {
        Tensor input = Tensor.from(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, 2, 1, 3);
        assertTensorGradients(input, value -> value.meanAxes(0, 2), Tensor.from(new float[]{0.7f}, 1),
                1e-3f, 1e-4f, 1e-3f);
        assertTensorGradients(input, value -> value.varianceAxes(new int[]{0, -1}, true, 1),
                Tensor.from(new float[]{0.7f}, 1, 1, 1), 1e-3f, 1e-4f, 1e-3f);
    }

    @Test
    void uniqueAxisMaximaMatchFiniteDifferencesAndTiesShareTheGradient() {
        Tensor input = Tensor.from(new float[]{-0.3f, 0.1f, 0.5f, 0.8f, 0.2f, -0.1f}, 2, 3);
        assertTensorGradients(input, value -> value.maximum(-1, false), Tensor.from(new float[]{2, 3}),
                1e-3f, 1e-4f, 1e-3f);
        Tensor ties = Tensor.from(new float[]{3, 3, 1, 2}, 2, 2).requiresGrad(true);
        ties.maximum(1, false).sumTensor().backward();
        assertArrayEquals(new float[]{0.5f, 0.5f, 0, 1}, ties.grad().toFloatArray());
    }

    @Test
    void compactVarianceUsesFloat32CentersAndLeafGradients() {
        Tensor input = Tensor.from(new float[]{0.7f, 1.3f, 2.1f}, DType.BFLOAT16).requiresGrad(true);
        float[] values = input.toFloatArray();
        float mean = (values[0] + values[1] + values[2]) / 3;
        input.variance(0, false).backward();
        assertArrayEquals(new float[]{(values[0] - mean) * (2.0f / 3),
                (values[1] - mean) * (2.0f / 3), (values[2] - mean) * (2.0f / 3)}, input.grad().toFloatArray());
        assertEquals(DType.FLOAT32, input.grad().dtype());
    }

    @Test
    void emptyAndUndefinedStatisticsHaveDefinedBackwardShapes() {
        Tensor empty = new Tensor(0, 3).requiresGrad(true);
        empty.maximum(1, false).sumTensor().backward();
        assertArrayEquals(new int[]{0, 3}, empty.grad().shape());
        Tensor singleton = Tensor.from(new float[]{1}).requiresGrad(true);
        singleton.variance(0, false, 1).backward();
        assertTrue(Float.isNaN(singleton.grad().get(0)));
        Tensor nan = Tensor.from(new float[]{Float.NaN, 2}).requiresGrad(true);
        nan.maxAxes().backward();
        assertTrue(Float.isNaN(nan.grad().get(0)));
        assertTrue(Float.isNaN(nan.grad().get(1)));
    }

    @Test
    void typedLegacyMatrixStatisticsRetainTheirShapesAndCorrectGradients() {
        Tensor input = Tensor.from(new float[]{1, 3}, DType.FLOAT16, 1, 2).requiresGrad(true);
        input.varianceAlongRows().backward();
        assertArrayEquals(new float[]{-1, 1}, input.grad().toFloatArray());
        input.zeroGrad();
        input.maxAlongRows().backward();
        assertArrayEquals(new float[]{0, 1}, input.grad().toFloatArray());
    }
}
