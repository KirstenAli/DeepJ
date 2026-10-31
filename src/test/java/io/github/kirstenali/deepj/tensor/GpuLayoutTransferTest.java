package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class GpuLayoutTransferTest {
    private final GpuStorageRuntime runtime = new GpuStorageRuntime();
    private final ComputeGraph graph = new ComputeGraph(runtime);

    @Test
    void stridedCopiesPreserveDtypeShapeAndRawBitsInLogicalOrder() {
        for (DType dtype : DType.values()) {
            Tensor source = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, dtype, 2, 3);
            Tensor view = Tensor.aliasStorage(source, LayoutTransforms.permute(source.layout(), 1, 0));
            Tensor copy = graph.copyToDevice(view);
            assertFalse(copy.memory().hasHostStorage());
            graph.flush();
            assertEquals(dtype, copy.dtype());
            assertArrayEquals(new int[]{3, 2}, copy.shape());
            assertTrue(graph.layouts.isEmpty());
            graph.materialize(copy);
            assertArrayEquals(new float[]{1, 4, 2, 5, 3, 6}, copy.toFloatArray());
        }
        graph.releaseAll();
    }

    @Test
    void typedCastsUseLayoutCommandsAndSameDtypeCastsRemainIdentity() {
        var cpu = new CpuBackend();
        for (DType sourceType : DType.values()) {
            Tensor source = Tensor.from(new float[]{1.5f, -2.25f, -0.0f, 4, 5.5f, 6.75f}, sourceType, 2, 3);
            Tensor view = Tensor.aliasStorage(source, LayoutTransforms.permute(source.layout(), 1, 0));
            assertSame(view, graph.castToDevice(view, sourceType));
            for (DType target : DType.values()) {
                Tensor output = graph.castToDevice(view, target);
                graph.flush();
                graph.materialize(output);
                assertEquals(target, output.dtype());
                assertArrayEquals(cpu.cast(view, target).toFloatArray(), output.toFloatArray());
            }
        }
        graph.releaseAll();
    }

    @Test
    void layoutCommandsKeepFixedBufferOffsetsAndCarryImmutableMetadataPerFlush() {
        Tensor source = Tensor.from(new float[]{1, 2, 3, 4}, 2, 2);
        Tensor view = Tensor.aliasStorage(source, LayoutTransforms.permute(source.layout(), 1, 0));
        Tensor output = graph.copyToDevice(view);
        int input = ((GpuBuffer) source.getGpuTag()).id, result = ((GpuBuffer) output.getGpuTag()).id;
        assertEquals(4, graph.cmdPos);
        assertEquals(ComputeGraph.OP_LAYOUT_TRANSFER, graph.cmdStream[0]);
        assertTrue(graph.isBufferReferencedByPendingOps(input));
        assertTrue(graph.isBufferReferencedByPendingOps(result));
        graph.flush();
        assertArrayEquals(new long[]{2, 4, 4, 0, 2, 2, 1, 2}, runtime.lastLayouts);
        assertTrue(graph.layouts.isEmpty());
        graph.releaseAll();
    }

    @Test
    void emptyTransfersNeedNoRuntimeSupportButUnsupportedRealWorkFailsBeforeAllocation() {
        var recording = new ComputeGraphTestSupport.RecordingRuntime();
        var unsupported = new ComputeGraph(recording);
        Tensor input = new Tensor(2, 3);
        assertThrows(UnsupportedOperationException.class, () -> unsupported.castToDevice(input, DType.FLOAT16));
        assertTrue(unsupported.pendingAllocs.isEmpty());
        Tensor empty = unsupported.castToDevice(new Tensor(2, 0, 3), DType.BFLOAT16);
        assertEquals(0, empty.numel());
        assertEquals(DType.BFLOAT16, empty.dtype());
        assertThrows(IllegalArgumentException.class, () -> graph.transferLayout(null, DType.FLOAT32));
        assertThrows(NullPointerException.class, () -> graph.castToDevice(input, null));
        assertThrows(UnsupportedOperationException.class, () -> recording.flushOps(new int[0], 0, new long[]{1}));
        assertThrows(NullPointerException.class, () -> recording.flushOps(new int[0], 0, null));
        recording.flushOps(new int[0], 0, new long[0]);
        assertEquals(1, recording.flushCalls.size());
    }
}
