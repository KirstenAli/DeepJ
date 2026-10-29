package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Random;

import static io.github.kirstenali.deepj.testing.NdMatmulReference.assertProduct;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class MetalPaddedBatchMatmulTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(DType.class)
    void paddedBatchStridesKeepEveryMatrixIndependent(DType dtype) {
        for (int columns : new int[]{39, 40}) {
            Tensor left = values(dtype, 3, 35, columns).slice(1, 1, 34, 1).slice(2, 1, 38, 1);
            assertProduct(left.transpose(1, 2), left, 3, 37, 37);
            assertProduct(left, left.transpose(1, 2), 3, 33, 33);
            Tensor skipped = left.slice(0, 0, 3, 2);
            assertProduct(skipped.transpose(1, 2), skipped, 2, 37, 37);
        }
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void onlyOnePaddedOperandStillUsesItsOwnBatchAddress(DType dtype) {
        Tensor left = values(dtype, 3, 35, 39).slice(1, 1, 34, 1).slice(2, 1, 38, 1);
        Tensor right = values(dtype, 3, 37, 9);
        assertProduct(left, right, 3, 33, 9);
        assertProduct(right.transpose(1, 2), left.transpose(1, 2), 3, 9, 33);
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void paddedSlicesScatterExactFloat32GradientsToTheirParents(DType dtype) {
        Tensor leftParent = values(dtype, 2, 5, 7).requiresGrad(true);
        Tensor rightParent = values(dtype, 2, 6, 5).requiresGrad(true);
        Tensor left = leftParent.slice(1, 1, 4, 1).slice(2, 1, 5, 1);
        Tensor right = rightParent.slice(1, 1, 5, 1).slice(2, 1, 4, 1);
        float[] expectedLeft = leftGradient(right.toFloatArray());
        float[] expectedRight = rightGradient(left.toFloatArray());
        left.matmul(right).sumTensor().backward();
        assertEquals(DType.FLOAT32, leftParent.grad().dtype());
        assertEquals(DType.FLOAT32, rightParent.grad().dtype());
        assertArrayEquals(expectedLeft, leftParent.grad().toFloatArray());
        assertArrayEquals(expectedRight, rightParent.grad().toFloatArray());
    }

    private static float[] leftGradient(float[] right) {
        float[] expected = new float[70];
        for (int index = 0; index < expected.length; index++) {
            int batch = index / 35, row = index % 35 / 7, inner = index % 7;
            if (row < 1 || row >= 4 || inner < 1 || inner >= 5) continue;
            for (int column = 0; column < 3; column++) {
                expected[index] += right[batch * 12 + (inner - 1) * 3 + column];
            }
        }
        return expected;
    }

    private static float[] rightGradient(float[] left) {
        float[] expected = new float[60];
        for (int index = 0; index < expected.length; index++) {
            int batch = index / 30, inner = index % 30 / 5, column = index % 5;
            if (inner < 1 || inner >= 5 || column < 1 || column >= 4) continue;
            for (int row = 0; row < 3; row++) {
                expected[index] += left[batch * 12 + row * 4 + inner - 1];
            }
        }
        return expected;
    }

    private static Tensor values(DType dtype, int batches, int rows, int columns) {
        float[] values = new float[batches * rows * columns];
        Random random = new Random(values.length + 20261028L);
        for (int index = 0; index < values.length; index++) {
            values[index] = random.nextInt(-7, 8) / 8.0f;
        }
        return Tensor.from(values, dtype, batches, rows, columns);
    }
}
