package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

import static io.github.kirstenali.deepj.testing.NdMatmulReference.assertProduct;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class MetalNdMatmulReferenceTest extends MetalDifferentialTestSupport {
    @Test
    void ranksTwoThroughSixBroadcastAndTransposeAgainstIndependentDoubleProducts() {
        Random random = new Random(2026102607L);
        for (int rank = 2; rank <= 6; rank++) {
            for (DType dtype : DType.values()) {
                checkRank(rank, dtype, random);
            }
        }
    }

    private static void checkRank(int rank, DType dtype, Random random) {
        int[] shape = new int[rank];
        Arrays.fill(shape, 2);
        shape[rank - 1] = 3;
        Tensor left = values(dtype, random, shape);
        int[] rightShape = new int[rank];
        Arrays.fill(rightShape, 2);
        rightShape[rank - 2] = 3;
        Tensor right = values(dtype, random, rightShape);
        int[] expected = shape.clone();
        expected[rank - 1] = 2;
        assertProduct(left, right, expected);
        assertProduct(left, right.transpose(rank - 2, rank - 1).contiguous()
                .transpose(rank - 2, rank - 1), expected);
        if (rank > 2) assertProduct(left, right.select(0, 0), expected);
    }

    @Test
    void bothStridedOperandsAndMixedDtypesMatchTheIndependentReference() {
        Random random = new Random(17);
        Tensor left = values(DType.FLOAT16, random, 2, 2, 6).slice(2, 0, 6, 2);
        Tensor right = values(DType.BFLOAT16, random, 3, 4).slice(1, 0, 4, 2);
        assertProduct(left, right, 2, 2, 2);
        assertProduct(left.to(DType.FLOAT32), right.to(DType.FLOAT32), 2, 2, 2);
        Tensor vector = values(DType.FLOAT32, random, 3);
        assertProduct(vector, right, 2);
        assertProduct(left, vector, 2, 2);
        assertProduct(vector, vector);
    }

    @Test
    void denseFloat32NdMatricesReuseTheExistingBatchedMpsNumerics() {
        Tensor a = values(DType.FLOAT32, new Random(45), 3, 7, 5);
        Tensor b = values(DType.FLOAT32, new Random(19), 3, 5, 9);
        Tensor legacy = metal.batchedMatmul(a.reshape(21, 5), b.reshape(15, 9), 3, false, false);
        assertArrayEquals(legacy.toFloatArray(), a.matmul(b).toFloatArray());
        Tensor transposed = a.transpose(1, 2);
        Tensor oldTranspose = metal.batchedMatmul(a.reshape(21, 5), a.reshape(21, 5), 3, true, false);
        assertArrayEquals(oldTranspose.toFloatArray(), transposed.matmul(a).toFloatArray());
    }

    @Test
    void repeatedMatrixAxesPaddedRowsAndOffsetsHaveIndependentProducts() {
        Random random = new Random(23);
        Tensor left = values(DType.FLOAT32, random, 5, 7).slice(0, 1, 5, 2).slice(1, 1, 4, 1);
        Tensor right = values(DType.FLOAT32, random, 3, 2);
        assertProduct(left, right, 2, 2);
        Tensor repeated = values(DType.FLOAT32, random, 2, 1).expand(2, 3);
        assertProduct(repeated, right, 2, 2);
        assertProduct(right.transpose(0, 1), repeated.transpose(0, 1), 2, 2);
    }

    @Test
    void largeRanksUseTheDescriptorStreamRatherThanAnInlineMetadataLimit() {
        int[] leftShape = new int[3000], rightShape = new int[3000], resultShape = new int[3000];
        Arrays.fill(leftShape, 1);
        Arrays.fill(rightShape, 1);
        Arrays.fill(resultShape, 1);
        leftShape[2999] = 3;
        rightShape[2998] = 3;
        Tensor left = Tensor.from(new float[]{1, 2, 3}, DType.FLOAT16, leftShape);
        Tensor right = Tensor.from(new float[]{2, 3, 4}, DType.FLOAT16, rightShape);
        Tensor result = left.matmul(right);
        assertArrayEquals(resultShape, result.shape());
        assertArrayEquals(new float[]{20}, result.toFloatArray());
    }

    private static Tensor values(DType dtype, Random random, int... shape) {
        int count = IntStream.of(shape).reduce(1, Math::multiplyExact);
        float[] values = new float[count];
        for (int index = 0; index < count; index++) { values[index] = random.nextInt(-7, 8) / 8.0f; }
        return Tensor.from(values, dtype, shape);
    }
}
