package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class GpuReductionRecordingTest {
    private final ComputeGraph graph = new ComputeGraph(new GpuStorageRuntime());

    @Test
    void reductionsGroupAxesInOriginalLogicalOrderWithoutCopyingValues() {
        Tensor input = new Tensor(DType.FLOAT16, 2, 3, 4);
        Tensor output = graph.layoutReduction(input, new int[]{0, -1}, false, 0, 0);
        assertArrayEquals(new int[]{3}, output.shape());
        assertEquals(DType.FLOAT16, output.dtype());
        assertEquals(ComputeGraph.OP_LAYOUT_REDUCE, graph.cmdStream[0]);
        assertEquals(1, graph.cmdStream[4]);
        assertArrayEquals(new long[]{3, 24, 24, 0, 3, 2, 4, 4, 12, 1}, graph.layouts.snapshot());
        assertEquals(2, graph.pendingAllocs.size());
        assertFalse(output.memory().hasHostStorage());
    }

    @Test
    void scalarTotalsAlwaysAccumulateIntoFloat32ButAxisOutputsRetainStorageDtype() {
        Tensor input = new Tensor(DType.BFLOAT16, 2, 3, 4);
        Tensor total = graph.layoutTotal(input, 4);
        assertEquals(0, total.rank());
        assertEquals(DType.FLOAT32, total.dtype());
        assertEquals(4, ((GpuBuffer) total.getGpuTag()).byteLength());
        Tensor mean = graph.layoutReduction(input, new int[]{2, 0}, true, 1, 0);
        assertArrayEquals(new int[]{1, 3, 1}, mean.shape());
        assertEquals(DType.BFLOAT16, mean.dtype());
    }

    @Test
    void emptyAndUndefinedStatisticsHaveExplicitResultsWithoutInputStorage() {
        Tensor empty = new Tensor(DType.FLOAT16, 2, 0, 3);
        Tensor sum = graph.layoutReduction(empty, new int[]{1}, false, 0, 0);
        graph.materialize(sum);
        assertArrayEquals(new float[6], sum.toFloatArray());
        Tensor mean = graph.layoutReduction(empty, new int[]{1}, false, 1, 0);
        graph.materialize(mean);
        for (float value : mean.toFloatArray()) { assertTrue(Float.isNaN(value)); }
        assertNull(empty.getGpuTag());
        assertEquals(0, graph.layoutReduction(empty, new int[]{0}, true, 0, 0).numel());
        assertEquals(0, graph.layoutSoftmax(empty, 1).numel());
        assertThrows(IllegalArgumentException.class,
                () -> graph.layoutReduction(empty, new int[]{1}, false, 3, 0));
        Tensor singleton = new Tensor(DType.FLOAT32, 1);
        graph.materialize(graph.layoutReduction(singleton, new int[]{0}, false, 2, 1));
        assertNull(singleton.getGpuTag());
    }

    @Test
    void softmaxAndMaximumAdjointsCarryAllBuffersAndLogicalOutputLayouts() {
        Tensor input = new Tensor(DType.FLOAT16, 2, 3, 4);
        Tensor output = graph.layoutSoftmax(input, 1);
        assertArrayEquals(input.shape(), output.shape());
        assertEquals(ComputeGraph.OP_LAYOUT_SOFTMAX, graph.cmdStream[0]);
        Tensor maximum = new Tensor(DType.FLOAT16, 2, 4), upstream = new Tensor(2, 4);
        Tensor gradient = graph.layoutMaximumGradient(input, maximum, upstream, new int[]{1}, false);
        assertArrayEquals(input.shape(), gradient.shape());
        assertEquals(DType.FLOAT32, gradient.dtype());
        assertEquals(ComputeGraph.OP_LAYOUT_MAXIMUM_BACKWARD, graph.cmdStream[6]);
        assertEquals(10, GraphExecution.getOpStride(ComputeGraph.OP_LAYOUT_MAXIMUM_BACKWARD));
    }

    @Test
    void invalidAxesTagsCorrectionsAndShapesFailBeforeValueAllocation() {
        Tensor input = new Tensor(DType.FLOAT16, 2, 3);
        assertThrows(IllegalArgumentException.class,
                () -> graph.layoutReduction(input, new int[]{0, -2}, false, 0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> graph.layoutReduction(input, new int[]{2}, false, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> graph.layoutReduction(input, new int[]{0}, false, -1, 0));
        assertThrows(IllegalArgumentException.class, () -> graph.layoutReduction(input, new int[]{0}, false, 0, 1));
        assertThrows(IllegalArgumentException.class, () -> graph.layoutReduction(input, new int[]{0}, false, 2, -1));
        assertThrows(IllegalArgumentException.class, () -> graph.layoutTotal(input, 1));
        assertThrows(IndexOutOfBoundsException.class, () -> graph.layoutSoftmax(input, 2));
        assertThrows(IllegalArgumentException.class,
                () -> graph.layoutMaximumGradient(input, new Tensor(2), new Tensor(2), new int[]{1}, false));
        assertTrue(graph.pendingAllocs.isEmpty());
    }
}
