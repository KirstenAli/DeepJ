package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertTensorGradients;
import static org.junit.jupiter.api.Assertions.*;

class MetalNdMatmulGradientTest extends MetalDifferentialTestSupport {
    @Test
    void stridedBatchAndBroadcastAdjointsMatchActualGpuFiniteDifferences() {
        Tensor left = Tensor.from(new float[]{0.2f, -0.3f, 0.1f, 0.4f, 0.5f, -0.2f, 0.3f, 0.6f}, 2, 2, 2);
        Tensor right = Tensor.from(new float[]{0.5f, -0.1f, 0.4f, 0.2f}, 1, 2, 2);
        Tensor upstream = Tensor.from(new float[]{0.1f, -0.4f, 0.2f, 0.3f, -0.1f, 0.5f, 0.4f, 0.2f}, 2, 2, 2);
        assertTensorGradients(left, value -> value.transpose(1, 2).matmul(right), upstream,
                1e-3f, 1e-4f, 1e-3f);
        assertTensorGradients(right, value -> left.transpose(1, 2).matmul(value), upstream,
                1e-3f, 1e-4f, 1e-3f);
    }

    @Test
    void vectorPromotionAndOffsetSlicesHaveFiniteDifferenceAdjoints() {
        Tensor vector = Tensor.from(new float[]{0.2f, -0.3f, 0.1f, 0.4f});
        Tensor matrix = Tensor.from(new float[]{0.1f, 0.3f, 0.5f, -0.2f}, 2, 2);
        Tensor upstream = Tensor.from(new float[]{0.5f, -0.2f});
        assertTensorGradients(vector, value -> value.slice(0, 0, 4, 2).matmul(matrix), upstream,
                1e-3f, 1e-4f, 1e-3f);
        assertTensorGradients(matrix, value -> value.matmul(vector.slice(0, 1, 4, 2)), upstream,
                1e-3f, 1e-4f, 1e-3f);
        assertTensorGradients(vector, value -> value.matmul(value), Tensor.from(new float[]{0.2f}, new int[0]),
                1e-3f, 1e-4f, 1e-3f);
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void compactMatrixAndBroadcastVectorLeavesReceiveFloat32Gradients(DType dtype) {
        Tensor left = Tensor.from(new float[]{1, 2, 3, 4}, dtype, 2, 2).requiresGrad(true);
        Tensor right = Tensor.from(new float[]{1, 2, 3, 4}, dtype, 2, 2).requiresGrad(true);
        left.matmul(right).sumTensor().backward();
        assertEquals(DType.FLOAT32, left.grad().dtype());
        assertEquals(DType.FLOAT32, right.grad().dtype());
        assertArrayEquals(new float[]{3, 7, 3, 7}, left.grad().toFloatArray());
        assertArrayEquals(new float[]{4, 4, 6, 6}, right.grad().toFloatArray());
        Tensor vector = Tensor.from(new float[]{2, 3}, dtype).requiresGrad(true);
        Tensor batches = Tensor.from(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, dtype, 2, 2, 2);
        batches.matmul(vector).sumTensor().backward();
        assertArrayEquals(new float[]{16, 20}, vector.grad().toFloatArray());
    }

    @Test
    void aliasMutationStillInvalidatesSavedMatrixValues() {
        Tensor source = Tensor.from(new float[]{1, 2, 3, 4}, 1, 2, 2).requiresGrad(true);
        Tensor output = source.matmul(source);
        try (var ignored = GradMode.noGrad()) { source.select(0, 0).set(0, 0, 5); }
        assertThrows(IllegalStateException.class, () -> output.sumTensor().backward());
    }

    @Test
    void emptyVectorAndBatchAdjointsPreserveOriginalShapes() {
        Tensor empty = new Tensor(0).requiresGrad(true);
        empty.matmul(empty).backward();
        assertArrayEquals(new int[]{0}, empty.grad().shape());
        Tensor left = new Tensor(0, 2, 3).requiresGrad(true), right = new Tensor(1, 3, 4).requiresGrad(true);
        left.matmul(right).sumTensor().backward();
        assertArrayEquals(new int[]{0, 2, 3}, left.grad().shape());
        assertArrayEquals(new float[12], right.grad().toFloatArray());
    }
}
