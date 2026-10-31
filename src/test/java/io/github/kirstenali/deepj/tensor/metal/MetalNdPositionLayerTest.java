package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.layers.*;
import io.github.kirstenali.deepj.layers.transformer.norm.LayerNorm1D;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import java.util.List;
import java.util.Random;
import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertTensorGradients;
import static org.junit.jupiter.api.Assertions.*;

class MetalNdPositionLayerTest extends MetalDifferentialTestSupport {
    @Test
    void projectionsAndNormLayersKeepExactOldMatrixForwardAndBackwardBitsForNdActivations() {
        for (Layer layer : List.of(new Linear(3, 2, new Random(5)), new BiasFreeLinear(3, 2, new Random(5)),
                new TiedEmbeddingProjection(new Parameter(Tensor.from(values(), 2, 3))),
                new LayerNorm1D(3), new RMSNorm1D(3))) {
            checkMatrix(layer);
        }
    }

    private static void checkMatrix(Layer layer) {
        Tensor input = Tensor.from(values(), 2, 1, 3).requiresGrad(true);
        Tensor actual = layer.forward(input);
        actual.sumTensor().backward();
        float[] gradient = input.grad().toFloatArray();
        List<Tensor> saved = layer.parameters().stream().map(p -> p.grad().detach()).toList();
        layer.parameters().forEach(Parameter::zeroGrad);
        Tensor matrix = input.detach().reshape(2, 3).requiresGrad(true);
        Tensor expected = layer.forward(matrix);
        expected.sumTensor().backward();
        assertArrayEquals(expected.toFloatArray(), actual.toFloatArray());
        assertArrayEquals(matrix.grad().toFloatArray(), gradient);
        for (int index = 0; index < saved.size(); index++) {
            assertArrayEquals(saved.get(index).toFloatArray(), layer.parameters().get(index).grad().toFloatArray());
        }
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void typedNdProjectionsHaveTrueShapesAndFloat32InputWeightBiasAdjoints(DType dtype) {
        Tensor input = metal.copy(Tensor.from(values(), dtype, 2, 1, 3)).requiresGrad(true);
        Tensor weight = Tensor.from(new float[]{1, 0.5f, -1, 0, 2, -0.5f}, dtype, 3, 2).requiresGrad(true);
        Tensor bias = Tensor.from(new float[]{0.25f, -0.5f}, dtype, 1, 2).requiresGrad(true);
        Tensor result = ProjectionOps.affine(input, weight, bias);
        result.sumTensor().backward();
        assertArrayEquals(new int[]{2, 1, 2}, result.shape());
        assertEquals(dtype, result.dtype());
        assertEquals(DType.FLOAT32, input.grad().dtype());
        assertEquals(DType.FLOAT32, weight.grad().dtype());
        assertArrayEquals(new float[]{2, 2}, bias.grad().toFloatArray());
        assertFalse(TensorTestStorage.hasHostStorage(input));
    }

    @Test
    void ndProjectionUsesActualGpuFiniteDifferenceInputAndParameterGradients() {
        Tensor input = Tensor.from(values(), 2, 1, 3), weight = Tensor.from(values(), 3, 2);
        Tensor bias = Tensor.from(new float[]{0.1f, -0.2f}, 1, 2), upstream = metal.zeros(DType.FLOAT32, 2, 1, 2)
                .addScalar(1);
        assertTensorGradients(input, x -> ProjectionOps.affine(x, weight, bias), upstream, 1e-3f, 3e-4f, 2e-3f);
        assertTensorGradients(weight, w -> ProjectionOps.affine(input, w, bias), upstream, 1e-3f, 3e-4f, 2e-3f);
        assertTensorGradients(bias, b -> ProjectionOps.affine(input, weight, b), upstream, 1e-3f, 3e-4f, 2e-3f);
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void legacyMaximumRejectsEmptyReductionAndPreservesNanAndZeroSign(DType dtype) {
        assertThrows(IllegalArgumentException.class, () -> metal.maxAlongRows(new Tensor(dtype, 2, 0)));
        assertArrayEquals(new int[]{0, 1}, metal.maxAlongRows(new Tensor(dtype, 0, 2)).shape());
        Tensor matrix = Tensor.from(new float[]{0, -0.0f, 1, Float.NaN, -0.0f, -0.0f}, dtype, 3, 2);
        assertArrayEquals(cpu.maxAlongRows(matrix).toFloatArray(), metal.maxAlongRows(matrix).toFloatArray());
    }

    private static float[] values() { return new float[]{0.5f, -0.25f, 1.5f, 2, -1, 0.75f}; }
}
