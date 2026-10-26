package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class GpuStorageCopyTest {
    private final GpuStorageRuntime runtime = new GpuStorageRuntime();
    private final ComputeGraph graph = new ComputeGraph(runtime);

    @Test
    void denseNdAndOffsetCopiesPreserveShapeAndStorageWithoutDownloadingTheSource() {
        for (DType dtype : DType.values()) { assertDenseCopies(dtype); }
        graph.releaseAll();
    }

    private void assertDenseCopies(DType dtype) {
        int before = runtime.downloads;
        Tensor source = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, dtype, 2, 1, 3);
        Tensor copy = graph.copyToDevice(source);
        assertArrayEquals(source.shape(), copy.shape());
        assertFalse(copy.memory().hasHostStorage());
        Tensor view = Tensor.aliasStorage(source, LayoutTransforms.select(source.layout(), 0, 1));
        Tensor offsetCopy = graph.copyToDevice(view);
        graph.flush();
        assertEquals(before, runtime.downloads);
        graph.materialize(copy);
        graph.materialize(offsetCopy);
        assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6}, copy.toFloatArray());
        assertArrayEquals(new float[]{4, 5, 6}, offsetCopy.toFloatArray());
        assertEquals(dtype, offsetCopy.dtype());
    }

    @Test
    void anExistingDeviceOutputCopyDoesNotMaterializeOrReuploadItsSource() {
        Tensor source = graph.createOutputTensor(graph.newOutputBuffer(1, 3));
        graph.flush();
        int inputId = ((GpuBuffer) source.getGpuTag()).id;
        runtime.uploadBuffer(inputId, new float[]{-0.0f, 2, Float.intBitsToFloat(0x7fc00123)});
        Tensor copy = graph.copyToDevice(source);
        assertNull(source.rawData());
        assertNull(copy.rawData());
        assertTrue(graph.pendingUploadData.isEmpty());
        graph.materialize(copy);
        assertEquals(1, runtime.downloads);
        assertNull(source.rawData());
        assertEquals(0x80000000, Float.floatToRawIntBits(copy.get(0, 0)));
        assertEquals(0x7fc00123, Float.floatToRawIntBits(copy.get(0, 2)));
        graph.releaseAll();
    }

    @Test
    void emptyCopiesNeedNoAllocationOrOperation() {
        Tensor source = new Tensor(DType.FLOAT16, 2, 0, 3);
        Tensor copy = graph.copyToDevice(source);
        assertArrayEquals(source.shape(), copy.shape());
        assertEquals(DType.FLOAT16, copy.dtype());
        assertEquals(0, copy.numel());
        assertFalse(copy.memory().hasHostStorage());
        assertTrue(graph.pendingAllocs.isEmpty());
        assertTrue(graph.isEmpty());
        Tensor view = Tensor.aliasStorage(source, LayoutTransforms.permute(source.layout(), 2, 1, 0));
        assertArrayEquals(view.shape(), graph.copyToDevice(view).shape());
    }

    @Test
    void typedCopyCommandIncludesPhysicalOffsetsAndParticipatesInLifetimeMetadata() {
        GpuBuffer input = graph.newOutputBuffer(1, 7, DType.BFLOAT16);
        GpuBuffer output = graph.newOutputBuffer(1, 8, DType.BFLOAT16);
        graph.recordStorageCopy(input, output, 1, 3, 5);
        assertArrayEquals(new int[]{ComputeGraph.OP_STORAGE_COPY, input.id, output.id, 1, 3, 5},
                Arrays.copyOf(graph.cmdStream, graph.cmdPos));
        assertTrue(graph.isBufferReferencedByPendingOps(input.id));
        assertTrue(graph.isBufferReferencedByPendingOps(output.id));
        assertEquals(6, GraphExecution.getOpStride(ComputeGraph.OP_STORAGE_COPY));
        graph.releaseAll();
    }

    @Test
    void copyValidationRejectsBadRangesAndFormatsBeforeRecordingAnything() {
        GpuBuffer input = graph.newOutputBuffer(1, 4, DType.FLOAT16);
        GpuBuffer output = graph.newOutputBuffer(1, 4, DType.FLOAT16);
        invalid(input, output, -1, 0, 1);
        invalid(input, output, 0, -1, 1);
        invalid(input, output, 0, 0, 0);
        invalid(input, output, 1, 0, 4);
        invalid(input, output, 0, 1, 4);
        invalid(input, output, Integer.MAX_VALUE, 0, Integer.MAX_VALUE);
        invalid(input, input, 0, 0, 1);
        invalid(input, graph.newOutputBuffer(1, 4), 0, 0, 1);
        assertEquals(0, graph.cmdPos);
        graph.releaseAll();
    }

    private void invalid(GpuBuffer input, GpuBuffer output, int source, int target, int count) {
        assertThrows(IllegalArgumentException.class,
                () -> graph.recordStorageCopy(input, output, source, target, count));
    }

    @Test
    void staleOrForeignBuffersCannotEnterACopyCommand() {
        GpuBuffer input = graph.newOutputBuffer(1, 1), output = graph.newOutputBuffer(1, 1);
        var other = new ComputeGraph(runtime);
        GpuBuffer foreign = other.newOutputBuffer(1, 1);
        invalid(foreign, output, 0, 0, 1);
        invalid(input, foreign, 0, 0, 1);
        other.releaseAll();
        assertThrows(IllegalStateException.class, () -> graph.recordStorageCopy(foreign, output, 0, 0, 1));
        assertEquals(0, graph.cmdPos);
        graph.releaseAll();
    }
}
