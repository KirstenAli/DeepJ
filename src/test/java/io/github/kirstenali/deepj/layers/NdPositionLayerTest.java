package io.github.kirstenali.deepj.layers;

import io.github.kirstenali.deepj.layers.transformer.norm.LayerNorm1D;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.Random;
import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertTensorGradients;
import static org.junit.jupiter.api.Assertions.*;

class NdPositionLayerTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void projectionsPreserveBatchTimeAndEveryLeadingAxisWithExactMatrixValuesAndGradients() {
        for (Layer layer : List.of(new Linear(3, 2, new Random(9)), new BiasFreeLinear(3, 2, new Random(9)),
                new TiedEmbeddingProjection(new Parameter(Tensor.from(values(), 2, 3))))) {
            for (int[] shape : shapes()) { compare(layer, shape); }
        }
    }

    private static void compare(Layer layer, int[] shape) {
        Tensor input = Tensor.from(values(), shape).requiresGrad(true);
        Tensor output = layer.forward(input), matrix = layer.forward(input.detach().reshape(2, 3));
        assertArrayEquals(matrix.toFloatArray(), output.toFloatArray());
        assertEquals(shape.length, output.rank());
        output.sumTensor().backward();
        float[] gradient = input.grad().toFloatArray();
        List<Tensor> parameters = layer.parameters().stream().map(p -> p.grad().detach()).toList();
        layer.parameters().forEach(Parameter::zeroGrad);
        Tensor reference = input.detach().reshape(2, 3).requiresGrad(true);
        layer.forward(reference).sumTensor().backward();
        assertArrayEquals(reference.grad().toFloatArray(), gradient);
        for (int index = 0; index < parameters.size(); index++) {
            assertArrayEquals(parameters.get(index).toFloatArray(),
                    layer.parameters().get(index).grad().toFloatArray());
        }
        layer.parameters().forEach(Parameter::zeroGrad);
    }

    @Test
    void positionwiseNormLayersKeepTheOriginalMatrixArithmeticAndParameterAdjoints() {
        for (Layer norm : List.of(new LayerNorm1D(3), new RMSNorm1D(3))) {
            for (int[] shape : shapes()) { compare(norm, shape); }
        }
    }

    @Test
    void vectorsEmptyPositionsAndStridesHaveHonestOutputShapes() {
        Linear layer = new Linear(3, 2, new Random(1));
        assertArrayEquals(new int[]{2}, layer.forward(Tensor.from(new float[]{1, 2, 3})).shape());
        assertArrayEquals(new int[]{0, 2, 2}, layer.forward(new Tensor(0, 2, 3)).shape());
        Tensor input = Tensor.from(values(), 3, 1, 2).permute(2, 1, 0);
        assertArrayEquals(layer.forward(input.contiguous()).toFloatArray(), layer.forward(input).toFloatArray());
        for (Layer norm : List.of(new LayerNorm1D(3), new RMSNorm1D(3))) {
            assertArrayEquals(new int[]{0, 2, 3}, norm.forward(new Tensor(0, 2, 3)).shape());
            assertArrayEquals(new int[]{3}, norm.forward(Tensor.from(new float[]{1, 2, 3})).shape());
        }
    }

    @Test
    void explicitCompactAffineAndTiedProjectionUseRealDtypesAndFloat32Adjoints() {
        for (DType dtype : DType.values()) { typedProjection(dtype); }
    }

    private static void typedProjection(DType dtype) {
        Tensor input = Tensor.from(values(), dtype, 2, 1, 3).requiresGrad(true);
        Tensor weight = Tensor.from(new float[]{1, 0.5f, -1, 0, 2, -0.5f}, dtype, 3, 2).requiresGrad(true);
        Tensor bias = Tensor.from(new float[]{0.25f, -0.5f}, dtype, 1, 2).requiresGrad(true);
        Tensor output = ProjectionOps.affine(input, weight, bias);
        assertEquals(dtype, output.dtype());
        assertEquals(4L * dtype.bytes(), output.storageBytes());
        output.sumTensor().backward();
        assertEquals(DType.FLOAT32, input.grad().dtype());
        assertEquals(DType.FLOAT32, weight.grad().dtype());
        assertArrayEquals(new float[]{2, 2}, bias.grad().toFloatArray());
        assertArrayEquals(new int[]{2, 1, 2}, ProjectionOps.tied(input.detach(), weight.transpose(0, 1)).shape());
    }

    @Test
    void ndProjectionInputWeightAndBiasHaveFiniteDifferenceGradientsAndInvalidWidthsFail() {
        Tensor input = Tensor.from(values(), 2, 1, 3), weight = Tensor.from(values(), 3, 2);
        Tensor bias = Tensor.from(new float[]{0.1f, -0.2f}, 1, 2), upstream = new Tensor(2, 1, 2).addScalar(1);
        assertTensorGradients(input, x -> ProjectionOps.affine(x, weight, bias), upstream, 1e-3f, 3e-4f, 2e-3f);
        assertTensorGradients(weight, w -> ProjectionOps.affine(input, w, bias), upstream, 1e-3f, 3e-4f, 2e-3f);
        assertTensorGradients(bias, b -> ProjectionOps.affine(input, weight, b), upstream, 1e-3f, 3e-4f, 2e-3f);
        assertThrows(IllegalArgumentException.class, () -> ProjectionOps.linear(new Tensor(new int[0]), weight));
        assertThrows(IllegalArgumentException.class, () -> ProjectionOps.linear(new Tensor(2, 2), weight));
        assertThrows(IllegalArgumentException.class, () -> ProjectionOps.affine(input, weight, new Tensor(2)));
        assertThrows(IllegalArgumentException.class, () -> new LayerNorm1D(3).forward(new Tensor(2, 2)));
    }

    private static float[] values() { return new float[]{0.5f, -0.25f, 1.5f, 2, -1, 0.75f}; }
    private static int[][] shapes() { return new int[][]{{2, 3}, {1, 2, 3}, {2, 1, 1, 3}, {1, 1, 2, 1, 1, 3}}; }
}
