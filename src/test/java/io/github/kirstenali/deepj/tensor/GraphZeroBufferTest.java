package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import java.lang.ref.WeakReference;

import static org.junit.jupiter.api.Assertions.*;

class GraphZeroBufferTest extends ComputeGraphTestSupport {
    @Test
    void zeroInitializationUsesMetadataUntilFlushAndPreservesPortableRuntimes() {
        GpuBuffer buffer = graph.newZeroBuffer(3, 5);
        Tensor tensor = graph.createOutputTensor(buffer);
        assertNull(tensor.rawData());
        assertNull(graph.pendingUploadData.get(0));
        assertArrayEquals(new int[]{buffer.id, 15}, graph.pendingUploadIds.get(0));
        graph.flush();
        assertArrayEquals(new float[15], runtime.uploads.get(0).data());
        assertTrue(graph.pendingUploadIds.isEmpty());
        assertTrue(graph.pendingUploadData.isEmpty());
        assertEquals(buffer.id, runtime.uploads.get(0).bufId());
    }

    @Test
    void removingAnUnallocatedBufferRemovesItsInitialization() {
        GpuBuffer buffer = graph.newZeroBuffer(1, 7);
        graph.removePendingForId(buffer.id);
        graph.flush();
        assertTrue(runtime.uploads.isEmpty());
    }

    @Test
    void zeroInputsCannotAliasRetiredScratchStorage() {
        GpuBuffer zero = graph.newZeroBuffer(1, 4);
        GpuBuffer first = graph.newOutputBuffer(1, 4), second = graph.newOutputBuffer(1, 4);
        graph.recordUnary(ComputeGraph.OP_NEG, zero, first);
        graph.recordUnary(ComputeGraph.OP_NEG, first, second);
        graph.bufIdToMemory.put(zero.id, new WeakReference<>(null));
        graph.bufIdToMemory.put(first.id, new WeakReference<>(null));
        assertTrue(GraphBufferReuse.prepare(graph).allocations().stream()
                .anyMatch(allocation -> allocation[0] == zero.id));
    }

    @Test
    void portableZeroFallbackRejectsInvalidIdsAndLengths() {
        assertThrows(IllegalArgumentException.class, () -> runtime.zeroBuffer(-1, 1));
        assertThrows(IllegalArgumentException.class, () -> runtime.zeroBuffer(1, 0));
        assertThrows(IllegalArgumentException.class, () -> runtime.zeroBuffer(1, -1));
        assertTrue(runtime.uploads.isEmpty());
    }
}
