package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertTensorGradients;
import static org.junit.jupiter.api.Assertions.*;

class NdAutogradTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void scalarThroughRankSixGradientsMatchTheIndependentQuadraticDerivative() {
        for (int rank = 0; rank <= 6; rank++) {
            int[] shape = new int[rank];
            java.util.Arrays.fill(shape, 2);
            float[] values = new float[1 << rank], expected = new float[values.length];
            for (int index = 0; index < values.length; index++) {
                values[index] = index * 0.01f - 0.3f;
                expected[index] = 2 * values[index];
            }
            Tensor input = Tensor.from(values, shape).requiresGrad(true);
            input.multiply(input).addScalar(1).sumTensor().backward();
            assertArrayEquals(expected, input.grad().toFloatArray());
            assertArrayEquals(shape, input.grad().shape());
        }
    }

    @Test
    void broadcastedLeavesAccumulateTheSumOfAllExpandedUses() {
        Tensor left = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, 2, 1, 3).requiresGrad(true);
        Tensor right = Tensor.from(new float[]{2, 3, 4}).requiresGrad(true);
        left.multiply(right).add(left).sumTensor().backward();
        assertArrayEquals(new float[]{3, 4, 5, 3, 4, 5}, left.grad().toFloatArray());
        assertArrayEquals(new float[]{5, 7, 9}, right.grad().toFloatArray());
        assertEquals(DType.FLOAT32, right.grad().dtype());
    }

    @Test
    void viewPermutationExpansionSliceAndCopyGradientsMatchFiniteDifferences() {
        Tensor input = Tensor.from(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, 2, 3);
        Tensor upstream = Tensor.from(new float[]{0.5f, -0.2f, 0.3f});
        assertTensorGradients(input, value -> value.view(2, 1, 3).expand(2, 4, 3).permute(1, 0, 2)
                        .slice(0, 1, 4, 2).contiguous().reshape(4, 3).select(0, -1),
                upstream, 1e-3f, 1e-4f, 1e-3f);
    }

    @Test
    void nonContiguousReshapeAndAxisSoftmaxGradientsMatchFiniteDifferences() {
        Tensor input = Tensor.from(new float[]{-0.3f, -0.1f, 0.2f, 0.1f, 0.3f, 0.5f}, 2, 3);
        Tensor upstream = Tensor.from(new float[]{0.2f, -0.1f, 0.3f, 0.2f, 0.4f, -0.3f}, 3, 2);
        assertTensorGradients(input, value -> value.transpose(0, 1).reshape(3, 2).softmax(0),
                upstream, 1e-3f, 1e-4f, 1e-3f);
    }

    @Test
    void reductionsRestoreAllReducedDimensionsAndEmptyBroadcastsHaveZeroGradients() {
        Tensor input = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, 2, 1, 3).requiresGrad(true);
        input.sum(0, false).sumAxes(-1).backward(Tensor.from(new float[]{2}, 1));
        assertArrayEquals(new float[]{2, 2, 2, 2, 2, 2}, input.grad().toFloatArray());
        Tensor singleton = Tensor.from(new float[]{1, 2, 3}, 1, 3).requiresGrad(true);
        singleton.add(new Tensor(0, 3)).sumTensor().backward();
        assertArrayEquals(new float[]{0, 0, 0}, singleton.grad().toFloatArray());
    }

    @Test
    void managedAliasMutationStillInvalidatesTheSavedGraph() {
        Tensor input = Tensor.from(new float[]{1, 2, 3, 4}, 2, 2).requiresGrad(true);
        Tensor output = input.multiply(input).sumTensor();
        try (var ignored = GradMode.noGrad()) { input.view(4).setAt(9, 0); }
        assertThrows(IllegalStateException.class, output::backward);
        assertThrows(IllegalStateException.class, () -> input.expand(2, 2).setAt(3, 0, 0));
    }

    @Test
    void dtypeConversionsHaveFloat32LeafGradientsAndDoNotEnableAutocast() {
        Tensor input = Tensor.from(new float[]{1.0003f, 2.0003f}).requiresGrad(true);
        Tensor output = input.to(DType.FLOAT16).to(DType.FLOAT32).sumTensor();
        output.backward();
        assertEquals(DType.FLOAT32, input.dtype());
        assertEquals(DType.FLOAT32, input.grad().dtype());
        assertArrayEquals(new float[]{1, 1}, input.grad().toFloatArray());
    }

    @Test
    void compactDivisionUsesFloat32DerivativeArithmetic() {
        for (DType dtype : new DType[]{DType.FLOAT16, DType.BFLOAT16}) {
            Tensor left = Tensor.from(new float[]{1.5f, 2.5f}, dtype).requiresGrad(true);
            Tensor right = Tensor.from(new float[]{0.7f, 1.3f}, dtype).requiresGrad(true);
            left.divide(right).sumTensor().backward();
            float[] a = left.toFloatArray(), b = right.toFloatArray();
            assertArrayEquals(new float[]{1 / b[0], 1 / b[1]}, left.grad().toFloatArray(), 1e-6f);
            assertArrayEquals(new float[]{-a[0] / (b[0] * b[0]), -a[1] / (b[1] * b[1])},
                    right.grad().toFloatArray(), 1e-6f);
            assertEquals(DType.FLOAT32, right.grad().dtype());
        }
    }

    @Test
    void compactSavedActivationsDoNotRequantizeDerivativeIntermediates() {
        Tensor input = Tensor.from(new float[]{0.7f, 1.3f}, DType.FLOAT16).requiresGrad(true);
        Tensor output = input.tanhActivation();
        float[] values = output.toFloatArray();
        output.sumTensor().backward();
        assertArrayEquals(new float[]{1 - values[0] * values[0], 1 - values[1] * values[1]},
                input.grad().toFloatArray(), 1e-6f);
        input.zeroGrad();
        input.pow(0).sumTensor().backward();
        assertArrayEquals(new float[]{0, 0}, input.grad().toFloatArray());
    }

    @Test
    void gradientEdgesOwnDefensiveShapeSnapshots() {
        Tensor input = new Tensor(2, 3, 4).requiresGrad(true);
        GradientEdge edge = new GradientEdge(input);
        int[] shape = edge.shape();
        shape[0] = 9;
        edge.requireShape(new Tensor(2, 3, 4));
        assertThrows(IllegalArgumentException.class, () -> edge.requireShape(new Tensor(24)));
    }
}
