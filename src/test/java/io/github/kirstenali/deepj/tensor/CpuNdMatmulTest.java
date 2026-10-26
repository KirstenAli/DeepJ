package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static io.github.kirstenali.deepj.testing.NdMatmulReference.assertBroadcastFixture;

class CpuNdMatmulTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void vectorPromotionMatchesDotMatrixVectorAndVectorMatrixDefinitions() {
        Tensor vector = Tensor.from(new float[]{1, 2, 3});
        assertEquals(32, vector.matmul(Tensor.from(new float[]{4, 5, 6})).item());
        Tensor matrix = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        assertArrayEquals(new float[]{14, 32}, matrix.matmul(vector).toFloatArray());
        assertArrayEquals(new float[]{22, 28}, vector.matmul(matrix.reshape(3, 2)).toFloatArray());
        assertEquals(0, vector.matmul(vector).rank());
    }

    @Test
    void multipleBatchAxesBroadcastWithoutCrossingExampleBoundaries() {
        assertBroadcastFixture(DType.FLOAT32);
    }

    @Test
    void stridedMatricesAndVectorsAreReadWithoutHostContiguousCopies() {
        Tensor vector = Tensor.from(new float[]{1, 9, 2, 9, 3, 9}).slice(0, 0, 6, 2);
        assertFalse(vector.isContiguous());
        assertEquals(14, vector.matmul(Tensor.from(new float[]{1, 2, 3})).item());
        Tensor matrix = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, 2, 3).permute(1, 0);
        assertArrayEquals(new float[]{1, 8, 2, 10, 3, 12},
                matrix.matmul(Tensor.from2D(new float[][]{{1, 0}, {0, 2}})).toFloatArray());
    }

    @Test
    void compactMatmulAccumulatesBeforeRoundingTheFinalResult() {
        Tensor left = Tensor.from(new float[]{2048, 1, 1}, DType.FLOAT16);
        Tensor right = Tensor.from(new float[]{1, 1, 1}, DType.FLOAT16);
        Tensor result = left.matmul(right);
        assertEquals(DType.FLOAT16, result.dtype());
        assertEquals(2050, result.item());
        assertEquals(2, result.storageBytes());
        assertEquals(DType.FLOAT32, left.matmul(right.to(DType.BFLOAT16)).dtype());
    }

    @Test
    void zeroInnerDimensionsAndEmptyBatchesHaveDefinedResults() {
        assertEquals(0, new Tensor(0).matmul(new Tensor(0)).item());
        assertArrayEquals(new float[6], new Tensor(2, 0).matmul(new Tensor(0, 3)).toFloatArray());
        assertArrayEquals(new int[]{0, 2, 4}, new Tensor(0, 2, 3).matmul(new Tensor(1, 3, 4)).shape());
        assertThrows(IllegalArgumentException.class, () -> new Tensor().matmul(new Tensor(1)));
        assertThrows(IllegalArgumentException.class, () -> new Tensor(3).matmul(new Tensor(4)));
        assertThrows(IllegalArgumentException.class, () -> new Tensor(2, 2, 3).matmul(new Tensor(3, 3, 4)));
    }

    @Test
    void singletonInsertionPreservesStorageAndValidatesAxes() {
        Tensor vector = Tensor.from(new float[]{1, 2, 3});
        Tensor column = vector.unsqueeze(-1);
        assertArrayEquals(new int[]{3, 1}, column.shape());
        column.set(1, 0, 8);
        assertEquals(8, vector.get(1));
        assertThrows(IndexOutOfBoundsException.class, () -> vector.unsqueeze(2));
    }
}
