package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class GraphHostCopiesTest {
    @Test
    void hostCopiesUploadOnlyTheLogicalRangeToOneIndependentAllocation() {
        for (DType dtype : DType.values()) { assertSingleAllocation(dtype); }
    }

    private static void assertSingleAllocation(DType dtype) {
        var runtime = new GpuStorageRuntime();
        var graph = new ComputeGraph(runtime);
        Tensor source = Tensor.from(new float[]{7, 9, -0.0f, 3}, dtype, 2, 2);
        Tensor view = Tensor.aliasStorage(source, LayoutTransforms.select(source.layout(), 0, 1));
        Tensor copy = graph.copyToDevice(view);
        assertEquals(1, graph.pendingAllocs.size());
        assertEquals(0, graph.cmdPos);
        assertNull(source.getGpuTag());
        source.set(1, 0, 99);
        graph.materialize(copy);
        assertArrayEquals(new float[]{-0.0f, 3}, copy.toFloatArray());
        assertEquals(dtype, copy.dtype());
        graph.releaseAll();
    }

    @Test
    void aFullPendingSnapshotBatchFlushesBeforeAnotherCopyIsQueued() {
        var runtime = new CountingRuntime();
        var graph = new ComputeGraph(runtime);
        // Repeated references simulate queued bytes without allocating a large test heap.
        float[] block = new float[1024];
        for (int index = 0; index < GraphHostCopies.SNAPSHOT_LIMIT / 4096; index++) {
            graph.scheduleUpload(7, block);
        }
        Tensor result = graph.copyToDevice(Tensor.ones(1, 2));
        assertEquals(GraphHostCopies.SNAPSHOT_LIMIT / 4096, runtime.uploads);
        assertEquals(1, graph.pendingUploadData.size());
        assertFalse(result.memory().hasHostStorage());
        graph.releaseAll();
    }

    @Test
    void typedSnapshotsAndZeroInitializersAreCountedWithoutFloatShadows() {
        var runtime = new CountingRuntime();
        var graph = new ComputeGraph(runtime);
        short[] block = new short[1024];
        for (int index = 0; index < GraphHostCopies.SNAPSHOT_LIMIT / 2048; index++) {
            graph.pendingUploadIds.add(new int[]{7});
            graph.pendingUploadData.add(block);
        }
        graph.newZeroBuffer(1, 1);
        Tensor result = graph.copyToDevice(new Tensor(DType.BFLOAT16, 1));
        assertEquals(GraphHostCopies.SNAPSHOT_LIMIT / 2048, runtime.uploads);
        assertInstanceOf(short[].class, graph.pendingUploadData.get(0));
        assertFalse(result.memory().hasHostStorage());
        graph.releaseAll();
    }

    @Test
    void anOversizedSingleCopyIsSubmittedImmediatelyWithoutASecondSourceBuffer() {
        var runtime = new CountingRuntime();
        var graph = new ComputeGraph(runtime);
        Tensor source = new Tensor(DType.FLOAT16, (int) (GraphHostCopies.SNAPSHOT_LIMIT / 2 + 1));
        Tensor result = graph.copyToDevice(source);
        assertEquals(1, runtime.uploads);
        assertTrue(graph.pendingUploadData.isEmpty());
        assertNull(source.getGpuTag());
        assertTrue(((GpuBuffer) result.getGpuTag()).allocatedOnGpu);
        graph.releaseTemporary();
    }

    @Test
    void flushingAHostReuploadNeedsNeitherANewAllocationNorARecordedKernel() {
        var runtime = new GpuStorageRuntime();
        var graph = new ComputeGraph(runtime);
        Tensor source = Tensor.ones(1, 2);
        graph.ensureStorageBuffer(source);
        graph.flush();
        source.set(0, 1, 7);
        graph.ensureStorageBuffer(source);
        assertTrue(graph.pendingAllocs.isEmpty());
        assertEquals(0, graph.cmdPos);
        graph.flush();
        assertTrue(graph.pendingUploadData.isEmpty());
        Tensor copy = graph.copyToDevice(source);
        graph.materialize(copy);
        assertArrayEquals(new float[]{1, 7}, copy.toFloatArray());
        graph.releaseAll();
    }

    private static final class CountingRuntime implements GpuRuntime {
        long uploads;

        @Override public void allocBuffers(int[] ids, int[] sizes, int count) {}
        @Override public void allocTypedBuffers(int[] ids, int[] sizes, DType[] types, int count) {}
        @Override public void uploadBuffer(int id, float[] data) { uploads++; }
        @Override public void uploadBuffer16(int id, short[] data) { uploads++; }
        @Override public void zeroBuffer(int id, int count) {}
        @Override public void downloadBuffer(int id, float[] data) {}
        @Override public void downloadBuffer16(int id, short[] data) {}
        @Override public void releaseBuffers(int[] ids, int count) {}
        @Override public void flushOps(int[] commands, int length) {}
    }
}
