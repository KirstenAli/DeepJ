package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class GpuTypedStorageTest {
    private final GpuStorageRuntime runtime = new GpuStorageRuntime();
    private final ComputeGraph graph = new ComputeGraph(runtime);

    @Test
    void physicalAllocationUsesDtypeBytesNotLogicalShapeOrFloatCount() {
        for (DType dtype : DType.values()) {
            Tensor tensor = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, dtype, 1, 2, 3);
            GpuBuffer buffer = graph.ensureStorageBuffer(tensor);
            assertEquals(6, buffer.elementCount());
            assertEquals(dtype.storageBytes(6), buffer.byteLength());
            assertEquals(dtype, buffer.dtype);
            graph.flush();
            assertEquals(dtype, runtime.types.get(buffer.id));
            assertTrue(buffer.allocatedOnGpu);
            assertPhysicalBits(tensor, runtime.values.get(buffer.id));
        }
        graph.releaseAll();
        assertTrue(runtime.values.isEmpty());
    }

    @Test
    void stridedSubviewTransfersWholeSharedAllocationAndKeepsOneBinding() {
        Tensor tensor = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, DType.BFLOAT16, 2, 3);
        Tensor view = Tensor.aliasStorage(tensor, LayoutTransforms.permute(tensor.layout(), 1, 0));
        GpuBuffer buffer = graph.ensureStorageBuffer(view);
        assertSame(buffer, graph.ensureStorageBuffer(tensor));
        graph.flush();
        assertArrayEquals(tensor.memory().shorts(), (short[]) runtime.values.get(buffer.id));
        assertEquals(1, runtime.values.size());
        graph.releaseAll();
    }

    @Test
    void compactDeviceOutputIsLazyAndReleaseAllDownloadsOnlyOneSharedAllocation() {
        GpuBuffer buffer = graph.newOutputBuffer(1, 6, DType.FLOAT16);
        Tensor output = graph.createOutputTensor(buffer, 1, 2, 3);
        Tensor view = Tensor.aliasStorage(output, LayoutTransforms.permute(output.layout(), 0, 2, 1));
        assertFalse(output.memory().hasHostStorage());
        graph.flush();
        short[] bits = {0, (short) 0x8000, 1, (short) 0x7c00, (short) 0xfc00, (short) 0x7e51};
        runtime.uploadBuffer16(buffer.id, bits);
        graph.releaseAll();
        assertEquals(1, runtime.downloads);
        assertArrayEquals(bits, output.memory().shorts());
        assertSame(output.memory(), view.memory());
        assertNull(view.getGpuTag());
    }

    @Test
    void compactZerosInitializeTheWholeAllocationAndRetainedAliasesSurviveCleanup() {
        GpuBuffer buffer = graph.newZeroBuffer(1, 6, DType.BFLOAT16);
        Tensor output = graph.createOutputTensor(buffer, 2, 1, 3).retainDeviceBuffer();
        Tensor view = Tensor.aliasStorage(output);
        assertFalse(output.memory().hasHostStorage());
        graph.flush();
        assertArrayEquals(new short[6], (short[]) runtime.values.get(buffer.id));
        graph.releaseTemporary();
        assertSame(buffer, view.getGpuTag());
        output.releaseDeviceBufferRetention();
        graph.releaseAll();
        assertArrayEquals(new short[6], view.memory().shorts());
        assertNull(view.getGpuTag());
    }

    @Test
    void typedBindingsRejectMismatchedFormatAndShapeAndLegacyOpcodesRollBack() {
        GpuBuffer buffer = graph.newOutputBuffer(1, 6, DType.FLOAT16);
        assertThrows(IllegalArgumentException.class, () -> graph.createOutputTensor(buffer, 2, 2));
        assertThrows(IllegalArgumentException.class, () -> graph.bindTensorToBuffer(new Tensor(2, 3), buffer));
        GpuBuffer output = graph.newOutputBuffer(1, 6);
        assertThrows(UnsupportedOperationException.class, () -> graph.recordUnary(ComputeGraph.OP_NEG, buffer, output));
        assertEquals(0, graph.cmdPos);
        assertEquals(0, graph.opCount);
        graph.releaseAll();
    }

    @Test
    void compactTransfersPreserveEveryRawPatternAndQueuedSnapshots() {
        for (DType dtype : new DType[]{DType.FLOAT16, DType.BFLOAT16}) {
            Tensor source = new Tensor(dtype, 256, 256);
            short[] bits = source.memory().shorts();
            for (int index = 0; index < bits.length; index++) { bits[index] = (short) index; }
            GpuBuffer buffer = graph.ensureStorageBuffer(source);
            bits[17] = 0;
            graph.flush();
            assertEquals(17, ((short[]) runtime.values.get(buffer.id))[17]);
            bits[17] = 17;
            assertPhysicalBits(source, runtime.values.get(buffer.id));
            buffer.cpuStale = true;
            graph.materialize(source);
            assertPhysicalBits(source, runtime.values.get(buffer.id));
        }
        graph.releaseAll();
    }

    @Test
    void sourceMutationCannotChangeAnEarlierQueuedConsumer() {
        Tensor source = Tensor.from2D(new float[][]{{1, 2}});
        GpuBuffer input = graph.ensureGpuBuffer(source);
        GpuBuffer first = graph.newOutputBuffer(1, 2);
        Tensor result = graph.createOutputTensor(first);
        graph.recordUnary(ComputeGraph.OP_NEG, input, first);
        source.set(0, 0, 9);
        GpuBuffer changed = graph.ensureGpuBuffer(source);
        GpuBuffer second = graph.newOutputBuffer(1, 2);
        Tensor later = graph.createOutputTensor(second);
        graph.recordUnary(ComputeGraph.OP_NEG, changed, second);
        graph.materialize(later);
        graph.materialize(result);
        assertArrayEquals(new float[]{-1, -2}, result.toFloatArray());
        assertArrayEquals(new float[]{-9, -2}, later.toFloatArray());
        graph.releaseAll();
    }

    private static void assertPhysicalBits(Tensor source, Object actual) {
        if (source.dtype() == DType.FLOAT32) assertArrayEquals(source.memory().floats(), (float[]) actual);
        else assertArrayEquals(source.memory().shorts(), (short[]) actual);
    }
}
