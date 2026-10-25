package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class GpuBufferIdsTest extends ComputeGraphTestSupport {
    @Test
    void allocatingInAnOldGraphDoesNotMarkATransferredBufferAsAllocated() {
        Tensor value = Tensor.ones(1, 4);
        graph.ensureGpuBuffer(value);
        graph.flush();
        ComputeGraph next = new ComputeGraph(new RecordingRuntime());
        GpuBuffer transferred = next.ensureGpuBuffer(value);
        assertFalse(transferred.allocatedOnGpu);
        graph.newOutputBuffer(1, 4);
        graph.flush();
        assertFalse(transferred.allocatedOnGpu);
        next.flush();
        assertTrue(transferred.allocatedOnGpu);
        graph.releaseAll();
        assertSame(transferred, value.getGpuTag());
        next.releaseAll();
    }

    @Test
    void failedNativeReleaseDoesNotReturnItsLeaseToTheSharedPool() {
        FailingRuntime failing = new FailingRuntime();
        ComputeGraph other = new ComputeGraph(failing);
        GpuBuffer buffer = other.newOutputBuffer(1, 4);
        Tensor value = other.createOutputTensor(buffer);
        other.flush();
        failing.fail = true;
        assertThrows(IllegalStateException.class, other::releaseTemporary);
        assertSame(buffer, other.managedBuffers.get(buffer.id));
        java.lang.ref.Reference.reachabilityFence(value);
        GpuBuffer replacement = graph.newOutputBuffer(1, 4);
        assertNotEquals(buffer.id, replacement.id);
        failing.fail = false;
        other.releaseAll();
        assertTrue(other.managedBuffers.isEmpty());
        graph.releaseAll();
    }

    @Test
    void returnedIdsRemainUsableAtTheIntegerBoundary() {
        var pool = new GpuBufferIds(Integer.MAX_VALUE);
        int id = pool.acquire();
        assertEquals(Integer.MAX_VALUE, id);
        assertThrows(IllegalStateException.class, pool::acquire);
        pool.release(new int[]{id}, 1);
        assertEquals(id, pool.acquire());
        assertThrows(IllegalStateException.class, pool::acquire);
    }

    @Test
    void freeListGrowthAndEmptyReturnsPreserveUniqueLeases() {
        var pool = new GpuBufferIds(0);
        int[] ids = new int[100];
        for (int index = 0; index < ids.length; index++) {
            ids[index] = pool.acquire();
        }
        pool.release(ids, 0);
        pool.release(ids, ids.length);
        Set<Integer> distinct = new HashSet<>();
        for (int index = 0; index < ids.length; index++) {
            assertTrue(distinct.add(pool.acquire()));
        }
        assertEquals(100, distinct.size());
        assertEquals(100, pool.acquire());
    }

    @Test
    void repeatedGraphCleanupRecyclesScratchButNotRetainedInputs() {
        Tensor source = Tensor.ones(1, 4).retainDeviceBuffer();
        GpuBuffer input = graph.ensureGpuBuffer(source);
        Set<Integer> ids = new HashSet<>();
        for (int step = 0; step < 1000; step++) {
            GpuBuffer output = graph.newOutputBuffer(1, 4);
            graph.createOutputTensor(output);
            graph.recordUnary(ComputeGraph.OP_NEG, input, output);
            assertNotEquals(input.id, output.id);
            ids.add(output.id);
            graph.releaseTemporary();
        }
        assertEquals(1, ids.size());
        assertEquals(1, graph.managedBuffers.size());
        graph.releaseAll();
        assertTrue(graph.managedBuffers.isEmpty());
    }

    @Test
    void staleRawHandlesCannotBeReboundOrReadAfterIdReuse() {
        GpuBuffer stale = graph.newOutputBuffer(1, 4);
        graph.createOutputTensor(stale);
        graph.releaseAll();
        GpuBuffer replacement = graph.newOutputBuffer(1, 4);
        assertEquals(stale.id, replacement.id);
        assertThrows(IllegalStateException.class, () -> graph.createOutputTensor(stale));
        Tensor forged = Tensor.ones(1, 4);
        forged.setGpuTag(stale);
        assertThrows(IllegalStateException.class, () -> graph.ensureGpuBuffer(forged));
        assertThrows(IllegalStateException.class, () -> graph.materialize(forged));
        graph.releaseAll();
    }

    @Test
    void staleArgumentsRejectOnlyThePartialOperationAfterIdReuse() {
        GpuBuffer stale = graph.newOutputBuffer(1, 4);
        graph.releaseAll();
        GpuBuffer current = graph.newOutputBuffer(1, 4);
        assertEquals(stale.id, current.id);
        graph.recordUnary(ComputeGraph.OP_NEG, current, current);
        int validLength = graph.cmdPos;
        assertThrows(IllegalStateException.class,
                () -> graph.recordBinary(ComputeGraph.OP_ADD, current, stale, current));
        assertEquals(validLength, graph.cmdPos);
        assertEquals(1, graph.opCount);
        assertThrows(NullPointerException.class,
                () -> graph.recordUnary(ComputeGraph.OP_NEG, null, current));
        assertEquals(validLength, graph.cmdPos);
        graph.flush();
        assertEquals(validLength, runtime.flushCalls.get(0).cmdStreamLength());
        graph.releaseAll();
    }

    private static final class FailingRuntime extends RecordingRuntime {
        private boolean fail;

        @Override
        public void releaseBuffers(int[] ids, int count) {
            if (fail) throw new IllegalStateException("Native release failed");
            super.releaseBuffers(ids, count);
        }
    }
}
