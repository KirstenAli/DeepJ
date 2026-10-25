package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class LazyHostMutationTest extends ComputeGraphTestSupport {
    @Test
    void deviceUpdatesDiscardOnlyLazyHostShadowsAndRedownloadOnDemand() {
        GpuBuffer buffer = graph.newOutputBuffer(1, 2);
        Tensor tensor = graph.createOutputTensor(buffer);
        runtime.downloadResult = new float[]{1, 2};
        graph.materialize(tensor);
        float[] previous = tensor.data;
        graph.markUpdated(tensor);
        assertNull(tensor.data);
        assertArrayEquals(new float[]{1, 2}, previous);
        runtime.downloadResult = new float[]{3, 4};
        graph.materialize(tensor);
        assertArrayEquals(new float[]{3, 4}, tensor.data);
        assertNotSame(previous, tensor.data);
    }

    @Test
    void ordinaryCpuStorageIsNeverDiscardedByDeviceUpdates() {
        Tensor tensor = Tensor.ones(1, 2);
        float[] original = tensor.data;
        GpuBuffer buffer = graph.ensureGpuBuffer(tensor);
        graph.markUpdated(tensor);
        assertSame(original, tensor.data);
        assertTrue(buffer.cpuStale);
        assertFalse(buffer.needsUpload);
    }

    @Test
    void updatesRejectForeignReleasedAndNonDeviceStorage() {
        GpuBuffer buffer = graph.newOutputBuffer(1, 2);
        Tensor tensor = graph.createOutputTensor(buffer);
        graph.markUpdated(tensor);
        assertTrue(buffer.cpuStale);
        var other = new ComputeGraph(new RecordingRuntime());
        assertThrows(IllegalArgumentException.class, () -> other.markUpdated(tensor));
        assertThrows(IllegalArgumentException.class, () -> graph.markUpdated(Tensor.ones(1, 2)));
        graph.releaseAll();
        tensor.setGpuTag(buffer);
        assertThrows(IllegalStateException.class, () -> graph.markUpdated(tensor));
    }

    @Test
    void copiedDeviceStorageOwnsItsSnapshotWithoutAnEagerHostArray() {
        Tensor source = Tensor.from2D(new float[][]{{-0.0f, 3}});
        Tensor copy = graph.copyToDevice(source);
        assertNull(copy.data);
        source.data[1] = 9;
        graph.flush();
        assertArrayEquals(new float[]{-0.0f, 3}, runtime.uploads.get(0).data());
        assertThrows(IllegalArgumentException.class, () -> graph.copyToDevice(null));
        assertThrows(IllegalArgumentException.class, () -> graph.newZeroBuffer(0, 1));
    }
}
