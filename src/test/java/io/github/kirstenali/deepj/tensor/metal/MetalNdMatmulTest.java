package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorTestStorage;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.*;
import static io.github.kirstenali.deepj.testing.NdMatmulReference.*;

class MetalNdMatmulTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(DType.class)
    void vectorsStridesTransposesAndSingletonAxesHaveTheirTrueLogicalShapes(DType dtype) {
        Tensor vector = Tensor.from(new float[]{1, 9, 2, 9, 3, 9}, dtype).slice(0, 0, 6, 2);
        Tensor matrix = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, dtype, 2, 3);
        assertEquals(14, vector.matmul(Tensor.from(new float[]{1, 2, 3}, dtype)).item());
        assertArrayEquals(new float[]{14, 32}, matrix.matmul(vector).toFloatArray());
        assertArrayEquals(new float[]{22, 28}, vector.matmul(matrix.reshape(3, 2)).toFloatArray());
        Tensor transposed = matrix.permute(1, 0);
        assertArrayEquals(new float[]{1, 8, 2, 10, 3, 12},
                transposed.matmul(Tensor.from(new float[]{1, 0, 0, 2}, dtype, 2, 2)).toFloatArray());
        Tensor scalar = vector.matmul(vector);
        assertEquals(0, scalar.rank());
        assertEquals(dtype, scalar.dtype());
        assertEquals(dtype.bytes(), scalar.storageBytes());
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void partialAndBroadcastBatchesNeverCrossSequenceBoundaries(DType dtype) {
        assertBroadcastFixture(dtype);
        Tensor offset = broadcastLeft(dtype).slice(0, 1, 2, 1).reshape(2, 3);
        Tensor right = broadcastRight(dtype).select(1, 0).reshape(3, 2);
        assertArrayEquals(new float[]{16, 17, 22, 23}, offset.matmul(right).toFloatArray());
    }

    @Test
    void compactStorageAccumulatesInFloat32AndMixedTypesPromoteExplicitly() {
        Tensor half = Tensor.from(new float[]{2048, 1, 1}, DType.FLOAT16);
        Tensor ones = Tensor.from(new float[]{1, 1, 1}, DType.FLOAT16);
        assertEquals(2050, half.matmul(ones).item());
        Tensor mixed = half.matmul(ones.to(DType.BFLOAT16));
        assertEquals(DType.FLOAT32, mixed.dtype());
        assertEquals(2050, mixed.item());
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void gpuCreatedInputsAndOutputsKeepHostStorageLazy(DType dtype) {
        Tensor left = metal.copy(Tensor.from(new float[]{1, 2, 3, 4}, dtype, 1, 2, 2));
        Tensor right = metal.copy(Tensor.from(new float[]{1, 0, 0, 2}, dtype, 2, 2));
        Tensor output = left.matmul(right).addScalar(1);
        metal.graph.flush();
        assertFalse(TensorTestStorage.hasHostStorage(left));
        assertFalse(TensorTestStorage.hasHostStorage(right));
        assertFalse(TensorTestStorage.hasHostStorage(output));
        assertArrayEquals(new float[]{2, 5, 4, 9}, output.toFloatArray());
        assertEquals(4L * dtype.bytes(), output.storageBytes());
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void emptyBatchesAndZeroInnerDimensionsProduceDefinedTypedResults(DType dtype) {
        assertEquals(0, new Tensor(dtype, 0).matmul(new Tensor(dtype, 0)).item());
        assertArrayEquals(new float[6], new Tensor(dtype, 2, 0).matmul(new Tensor(dtype, 0, 3)).toFloatArray());
        Tensor empty = new Tensor(dtype, 0, 2, 3).matmul(new Tensor(dtype, 1, 3, 4));
        assertArrayEquals(new int[]{0, 2, 4}, empty.shape());
        assertEquals(dtype, empty.dtype());
        assertEquals(0, empty.storageBytes());
    }
}
