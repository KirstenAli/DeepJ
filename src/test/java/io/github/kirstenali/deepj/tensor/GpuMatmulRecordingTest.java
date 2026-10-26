package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class GpuMatmulRecordingTest {
    private final ComputeGraph graph = new ComputeGraph(new GpuStorageRuntime());

    @Test
    void promotedAndBroadcastLayoutsDescribeTheOriginalTypedAllocations() {
        Tensor left = new Tensor(DType.FLOAT16, 2, 1, 3, 4), right = new Tensor(DType.BFLOAT16, 5, 4, 6);
        Tensor output = graph.layoutMatmul(left, right);
        assertArrayEquals(new int[]{2, 5, 3, 6}, output.shape());
        assertEquals(DType.FLOAT32, output.dtype());
        assertEquals(3, graph.pendingAllocs.size());
        assertEquals(ComputeGraph.OP_LAYOUT_MATMUL, graph.cmdStream[0]);
        assertArrayEquals(new long[]{4, 120, 24, 0, 2, 5, 3, 4, 12, 0, 4, 1,
                4, 240, 120, 0, 2, 5, 4, 6, 0, 24, 6, 1}, graph.layouts.snapshot());
        assertFalse(output.memory().hasHostStorage());
        assertEquals(6, GraphExecution.getOpStride(ComputeGraph.OP_LAYOUT_MATMUL));
    }

    @Test
    void vectorDotsHaveScalarOutputsButMatrixDescriptors() {
        Tensor output = graph.layoutMatmul(new Tensor(DType.FLOAT16, 3), new Tensor(DType.FLOAT16, 3));
        assertEquals(0, output.rank());
        assertEquals(2, output.storageBytes());
        long[] words = graph.layouts.snapshot();
        assertEquals(2, words[0]);
        assertEquals(1, words[4]);
        assertEquals(3, words[5]);
        assertEquals(3, words[12]);
        assertEquals(1, words[13]);
    }

    @Test
    void emptyGeometryAndZeroInnerDimensionsNeverUploadInputs() {
        Tensor left = new Tensor(DType.FLOAT16, 2, 0), right = new Tensor(DType.FLOAT16, 0, 3);
        Tensor zero = graph.layoutMatmul(left, right);
        graph.materialize(zero);
        assertArrayEquals(new float[6], zero.toFloatArray());
        assertNull(left.getGpuTag());
        assertNull(right.getGpuTag());
        assertEquals(0, graph.layoutMatmul(new Tensor(0, 2, 3), new Tensor(1, 3, 4)).numel());
        assertEquals(0, graph.cmdPos);
    }

    @Test
    void invalidGeometryFailsBeforeAnyDeviceAllocation() {
        assertThrows(IllegalArgumentException.class, () -> graph.layoutMatmul(new Tensor(), new Tensor(1)));
        assertThrows(IllegalArgumentException.class, () -> graph.layoutMatmul(new Tensor(3), new Tensor(4)));
        assertThrows(IllegalArgumentException.class,
                () -> graph.layoutMatmul(new Tensor(2, 2, 3), new Tensor(3, 3, 4)));
        assertTrue(graph.pendingAllocs.isEmpty());
        assertTrue(graph.layouts.isEmpty());
    }
}
