package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class TensorNdStorageTest {
    @Test
    void scalarVectorAndHigherRanksUseOneTensorAndLogicalIndexing() {
        Tensor scalar = new Tensor();
        scalar.setAt(1.5f);
        assertEquals(0, scalar.rank());
        assertEquals(1.5f, scalar.get());
        scalar.setAt(2.5f);
        assertEquals(2.5f, scalar.item());
        Tensor vector = Tensor.from(new float[]{1, 2, 3}, 3);
        assertEquals(3, vector.get(-1));
        Tensor cube = new Tensor(2, 3, 4);
        cube.setAt(7.5f, 1, 2, 3);
        assertEquals(7.5f, cube.get(-1, -1, -1));
        assertEquals(24, cube.numel());
        assertEquals(0, cube.get(0, 0, 0));
    }

    @Test
    void emptyAxesPreserveRankAndDoNotAllocateLogicalValues() {
        Tensor tensor = new Tensor(DType.BFLOAT16, 2, 0, 3, 4);
        assertEquals(4, tensor.rank());
        assertEquals(0, tensor.numel());
        assertEquals(0, tensor.storageBytes());
        assertArrayEquals(new float[0], tensor.toFloatArray());
        assertThrows(IndexOutOfBoundsException.class, () -> tensor.get(0, 0, 0, 0));
        assertThrows(IllegalStateException.class, tensor::item);
    }

    @Test
    void geometryQueriesAreDefensiveAndMatrixHelpersNeverFlattenOtherRanks() {
        Tensor tensor = new Tensor(2, 3, 4);
        int[] shape = tensor.shape();
        long[] strides = tensor.strides();
        shape[0] = 99;
        strides[0] = 99;
        assertArrayEquals(new int[]{2, 3, 4}, tensor.shape());
        assertArrayEquals(new long[]{12, 4, 1}, tensor.strides());
        assertEquals(4, tensor.size(-1));
        assertThrows(IllegalArgumentException.class, tensor::matrixRows);
        assertThrows(IllegalArgumentException.class, tensor::matrixCols);
        assertEquals(2, new Tensor(2, 3).matrixRows());
        assertEquals(3, new Tensor(2, 3).matrixCols());
    }

    @Test
    void optionalTypesHaveGenuineCompactStorageWithExplicitFloatExports() {
        for (DType dtype : DType.values()) {
            Tensor tensor = Tensor.from(new float[]{1.5f, 2.5f, -3.5f, 0.0f}, dtype, 2, 2);
            assertEquals(dtype, tensor.dtype());
            assertEquals(4L * dtype.bytes(), tensor.storageBytes());
            assertArrayEquals(new float[]{1.5f, 2.5f, -3.5f, 0.0f}, tensor.toFloatArray());
            if (dtype != DType.FLOAT32) assertThrows(IllegalStateException.class, tensor::rawData);
        }
        assertEquals(DType.FLOAT32, new Tensor(2, 3, 4).dtype());
        assertArrayEquals(new int[]{2}, Tensor.from(new float[]{1.5f, 2.5f}).shape());
        assertEquals(DType.FLOAT16, Tensor.from(new float[]{1.5f, 2.5f}, DType.FLOAT16).dtype());
    }

    @Test
    void stridedViewsShareValuesAndVersionsWithoutUnsafeFlatArrayAccess() {
        Tensor root = Tensor.from(new float[]{0, 1, 2, 3, 4, 5}, 2, 3);
        Tensor view = Tensor.aliasStorage(root, LayoutTransforms.permute(root.layout(), 1, 0));
        assertArrayEquals(new int[]{3, 2}, view.shape());
        assertFalse(view.isContiguous());
        view.set(2, 1, 9);
        assertEquals(9, root.get(1, 2));
        assertEquals(root.gradientState().version(), view.gradientState().version());
        assertArrayEquals(new float[]{0, 3, 1, 4, 2, 9}, view.toFloatArray());
        assertThrows(IllegalStateException.class, view::rawData);
        assertThrows(IllegalStateException.class, view::data);
    }

    @Test
    void offsetViewAndDeepCopyAddressLogicalValuesOnly() {
        Tensor root = Tensor.from(new float[]{0, 1, 2, 3, 4, 5}, 2, 3);
        Tensor view = Tensor.aliasStorage(root, LayoutTransforms.select(root.layout(), 0, 1));
        assertEquals(3, view.storageOffset());
        assertEquals(24, view.storageBytes());
        Tensor copy = new Tensor(view);
        assertArrayEquals(new float[]{3, 4, 5}, copy.toFloatArray());
        assertEquals(12, copy.storageBytes());
        copy.setAt(10.0f, 0);
        assertEquals(3, view.get(0));
        assertEquals(10, copy.get(0));
        assertTrue(copy.isContiguous());
    }

    @Test
    void overlappingBroadcastReadsAreValidButWritesCannotChangeStorageOrVersion() {
        Tensor root = Tensor.from(new float[]{1, 2, 3}, 3);
        Tensor view = Tensor.aliasStorage(root, LayoutTransforms.expand(root.layout(), 2, 3));
        assertArrayEquals(new float[]{1, 2, 3, 1, 2, 3}, view.toFloatArray());
        assertThrows(IllegalStateException.class, () -> view.set(0, 0, 7));
        assertEquals(0, root.gradientState().version());
        assertEquals(1, root.get(0));
    }

    @Test
    void compactStridedCopyPreservesRawNanAndSignedZeroBits() {
        for (DType dtype : Arrays.asList(DType.FLOAT16, DType.BFLOAT16)) {
            Tensor source = new Tensor(dtype, 2, 2);
            short[] bits = {(short) 0x7c01, (short) 0x8000, (short) 0xff81, (short) 0x7fff};
            System.arraycopy(bits, 0, source.memory().shorts(), 0, bits.length);
            Tensor view = Tensor.aliasStorage(source, LayoutTransforms.permute(source.layout(), 1, 0));
            Tensor copy = new Tensor(view);
            assertEquals(dtype, copy.dtype());
            assertArrayEquals(new short[]{bits[0], bits[2], bits[1], bits[3]}, copy.memory().shorts());
            assertEquals(8, copy.storageBytes());
        }
    }

    @Test
    void invalidIndicesAndValueCountsCannotMutateTheTensor() {
        Tensor tensor = new Tensor(2, 3, 4);
        assertThrows(IllegalArgumentException.class, () -> tensor.get(1, 2));
        assertThrows(IndexOutOfBoundsException.class, () -> tensor.get(2, 0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> tensor.setAt(5.0f, 0, -4, 0));
        assertThrows(IllegalArgumentException.class, () -> Tensor.from(new float[]{1, 2}, 3));
        assertEquals(0, tensor.gradientState().version());
    }
}
