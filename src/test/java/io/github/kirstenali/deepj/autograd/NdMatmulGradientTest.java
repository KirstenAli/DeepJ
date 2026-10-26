package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertTensorGradients;
import static org.junit.jupiter.api.Assertions.*;

class NdMatmulGradientTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void vectorDotAndMatrixProductsRestoreUnpromotedGradientShapes() {
        Tensor left = Tensor.from(new float[]{1, 2, 3}).requiresGrad(true);
        Tensor right = Tensor.from(new float[]{4, 5, 6}).requiresGrad(true);
        left.matmul(right).backward();
        assertArrayEquals(new float[]{4, 5, 6}, left.grad().toFloatArray());
        assertArrayEquals(new float[]{1, 2, 3}, right.grad().toFloatArray());
        assertArrayEquals(new int[]{3}, right.grad().shape());
    }

    @Test
    void broadcastBatchedMatmulMatchesIndependentFiniteDifferencesForBothInputs() {
        Tensor left = Tensor.from(new float[]{0.1f, 0.2f, 0.3f, -0.1f, 0.4f, 0.6f,
                -0.2f, 0.7f, 0.5f, 0.4f, -0.3f, 0.2f}, 2, 2, 3);
        Tensor right = Tensor.from(new float[]{0.2f, -0.1f, 0.3f, 0.5f, 0.7f, -0.2f}, 1, 3, 2);
        Tensor upstream = Tensor.from(new float[]{0.1f, 0.2f, -0.3f, 0.4f, 0.5f, -0.6f, 0.7f, 0.8f}, 2, 2, 2);
        assertTensorGradients(left, value -> value.matmul(right), upstream, 1e-3f, 1e-4f, 1e-3f);
        assertTensorGradients(right, left::matmul, upstream, 1e-3f, 1e-4f, 1e-3f);
    }

    @Test
    void vectorMatrixAndMatrixVectorAdjointsMatchFiniteDifferences() {
        Tensor vector = Tensor.from(new float[]{0.1f, -0.2f, 0.3f});
        Tensor matrix = Tensor.from(new float[]{0.2f, 0.5f, -0.3f, 0.1f, 0.7f, 0.4f}, 3, 2);
        Tensor upstream = Tensor.from(new float[]{0.3f, -0.2f});
        assertTensorGradients(vector, value -> value.matmul(matrix), upstream, 1e-3f, 1e-4f, 1e-3f);
        assertTensorGradients(vector, value -> matrix.transpose(0, 1).matmul(value),
                upstream, 1e-3f, 1e-4f, 1e-3f);
        assertTensorGradients(matrix, vector::matmul, upstream, 1e-3f, 1e-4f, 1e-3f);
    }

    @Test
    void compactParametersStillReceiveFloat32MatrixGradients() {
        Tensor left = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, DType.FLOAT16, 2, 3).requiresGrad(true);
        Tensor right = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, DType.FLOAT16, 3, 2).requiresGrad(true);
        left.matmul(right).sumTensor().backward();
        assertEquals(DType.FLOAT32, left.grad().dtype());
        assertEquals(DType.FLOAT32, right.grad().dtype());
        assertArrayEquals(new float[]{3, 7, 11, 3, 7, 11}, left.grad().toFloatArray());
        assertArrayEquals(new float[]{5, 5, 7, 7, 9, 9}, right.grad().toFloatArray());
    }

    @Test
    void stridedVectorSliceScatterAndSingletonInsertionBackpropagate() {
        Tensor input = Tensor.from(new float[]{1, 9, 2, 9, 3, 9}).requiresGrad(true);
        input.slice(0, 0, 6, 2).matmul(Tensor.from(new float[]{1, 2, 3})).backward();
        assertArrayEquals(new float[]{1, 0, 2, 0, 3, 0}, input.grad().toFloatArray());
        input.zeroGrad();
        input.unsqueeze(0).sumTensor().backward();
        assertArrayEquals(new float[]{1, 1, 1, 1, 1, 1}, input.grad().toFloatArray());
    }
}
