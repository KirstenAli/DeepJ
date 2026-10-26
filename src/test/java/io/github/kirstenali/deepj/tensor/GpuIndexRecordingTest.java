package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class GpuIndexRecordingTest {
    private final ComputeGraph graph = new ComputeGraph(new GpuStorageRuntime());

    @Test
    void stableGroupsKeepEveryRepeatedIndexInOriginalOrderAndUseIntegerMetadata() {
        assertArrayEquals(new long[]{3, 2, 2, 0, 2, 2, 0, 0, 2, 3, 0, 2, 1},
                IndexGroups.describe(new int[]{2, 0, 2}));
        assertArrayEquals(new long[]{0, 0, 0}, IndexGroups.describe(new int[0]));
        assertEquals(16_777_217, IndexGroups.describe(new int[]{16_777_217})[2]);
    }

    @Test
    void indexSelectionRecordsIndependentTypedOutputWithoutValueDownloads() {
        Tensor input = new Tensor(DType.FLOAT16, 2, 3, 4);
        int[] indices = {2, 0, 2};
        Tensor output = graph.layoutIndexSelect(input, 1, indices);
        indices[0] = 99;
        assertArrayEquals(input.shape(), output.shape());
        assertEquals(DType.FLOAT16, output.dtype());
        assertEquals(2, graph.pendingAllocs.size());
        assertEquals(ComputeGraph.OP_LAYOUT_INDEX_SELECT, graph.cmdStream[0]);
        assertEquals(2, graph.layouts.snapshot()[12]);
        assertFalse(output.memory().hasHostStorage());
    }

    @Test
    void scatterAliasesTakeASnapshotAndBothBuffersParticipateInLifetimePlanning() {
        Tensor target = new Tensor(DType.BFLOAT16, 3);
        graph.layoutIndexAdd(target, 0, new int[]{2, 0, 1}, target);
        assertEquals(ComputeGraph.OP_STORAGE_COPY, graph.cmdStream[0]);
        assertEquals(ComputeGraph.OP_LAYOUT_INDEX_ADD, graph.cmdStream[6]);
        assertNotEquals(graph.cmdStream[7], graph.cmdStream[8]);
        assertEquals(2, graph.pendingAllocs.size());
        assertEquals(7, GraphExecution.getOpStride(ComputeGraph.OP_LAYOUT_INDEX_ADD));
    }

    @Test
    void emptyAndInvalidCasesNeverAllocateUnnecessaryDeviceValues() {
        Tensor input = new Tensor(DType.FLOAT16, 2, 0, 3);
        assertArrayEquals(new int[]{2, 0, 3}, graph.layoutIndexSelect(input, 1, new int[0]).shape());
        graph.layoutIndexAdd(input, 0, new int[]{1}, new Tensor(DType.FLOAT16, 1, 0, 3));
        assertThrows(IndexOutOfBoundsException.class, () -> graph.layoutIndexSelect(input, 1, new int[]{0}));
        assertThrows(IllegalArgumentException.class,
                () -> graph.layoutIndexAdd(input, 0, new int[]{1}, new Tensor(1, 0, 3)));
        assertThrows(IllegalArgumentException.class,
                () -> graph.layoutIndexGradient(new Tensor(2), new int[]{3}, 0, new int[]{0}));
        assertTrue(graph.pendingAllocs.isEmpty());
        assertTrue(graph.layouts.isEmpty());
    }
}
