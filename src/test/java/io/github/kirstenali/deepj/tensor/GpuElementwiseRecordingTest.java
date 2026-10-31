package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class GpuElementwiseRecordingTest {
    private final ComputeGraph graph = new ComputeGraph(new GpuStorageRuntime());

    @Test
    void broadcastsEncodeExpandedLogicalDescriptorsAndPromotedPhysicalBytes() {
        Tensor left = new Tensor(DType.FLOAT16, 2, 1, 3), right = new Tensor(DType.BFLOAT16, 4, 1);
        Tensor result = graph.layoutBinary(left, right, ComputeGraph.OP_ADD);
        assertArrayEquals(new int[]{2, 4, 3}, result.shape());
        assertEquals(DType.FLOAT32, result.dtype());
        assertEquals(96, ((GpuBuffer) result.getGpuTag()).byteLength());
        assertEquals(7, graph.cmdPos);
        assertEquals(ComputeGraph.OP_LAYOUT_BINARY, graph.cmdStream[0]);
        assertArrayEquals(new long[]{3, 24, 6, 0, 2, 4, 3, 3, 0, 1,
                3, 24, 4, 0, 2, 4, 3, 0, 1, 0}, graph.layouts.snapshot());
    }

    @Test
    void scalarCommandsCarryRawFloatBitsAndAnIndependentTypedOutput() {
        Tensor input = new Tensor(DType.FLOAT16, 2, 3, 4);
        Tensor output = graph.layoutUnary(input, ComputeGraph.OP_CLAMP, -0.0f, Float.POSITIVE_INFINITY);
        assertArrayEquals(input.shape(), output.shape());
        assertEquals(input.dtype(), output.dtype());
        assertNotSame(input.memory(), output.memory());
        assertEquals(Float.floatToRawIntBits(-0.0f), graph.cmdStream[5]);
        assertEquals(Float.floatToRawIntBits(Float.POSITIVE_INFINITY), graph.cmdStream[6]);
    }

    @Test
    void aliasedAssignmentsRecordAnIndependentSnapshotAndPreserveTheTargetBinding() {
        Tensor target = new Tensor(DType.BFLOAT16, 2, 3);
        Tensor alias = TensorViews.view(target, 2, 3);
        graph.assignToDevice(target, alias);
        assertSame(target.getGpuTag(), alias.getGpuTag());
        assertEquals(ComputeGraph.OP_STORAGE_COPY, graph.cmdStream[0]);
        assertEquals(ComputeGraph.OP_LAYOUT_ASSIGN, graph.cmdStream[6]);
        assertNotEquals(graph.cmdStream[7], graph.cmdStream[8]);
        assertTrue(((GpuBuffer) target.getGpuTag()).cpuStale);
        assertTrue(target.memory().hasHostStorage()); // An explicit CPU allocation keeps its host storage.
    }

    @Test
    void emptyOperationsNeedNoRuntimeAndInvalidOperationsFailBeforeAllocation() {
        var unsupported = new ComputeGraph(new ComputeGraphTestSupport.RecordingRuntime());
        Tensor empty = new Tensor(DType.FLOAT16, 0, 3), source = new Tensor(DType.FLOAT16, 1, 3);
        assertEquals(0, unsupported.layoutBinary(empty, source, ComputeGraph.OP_ADD).numel());
        assertEquals(0, unsupported.layoutUnary(empty, ComputeGraph.OP_NEG, 0, 0).numel());
        unsupported.assignToDevice(empty, empty);
        assertThrows(UnsupportedOperationException.class,
                () -> unsupported.layoutBinary(source, source, ComputeGraph.OP_ADD));
        assertThrows(UnsupportedOperationException.class,
                () -> unsupported.layoutUnary(source, ComputeGraph.OP_NEG, 0, 0));
        assertThrows(UnsupportedOperationException.class, () -> unsupported.assignToDevice(source, source));
        assertTrue(unsupported.pendingAllocs.isEmpty());
        assertThrows(IllegalArgumentException.class, () -> graph.layoutBinary(source, source, -1));
        assertThrows(IllegalArgumentException.class, () -> graph.layoutUnary(source, -1, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> graph.layoutBinary(null, source, ComputeGraph.OP_ADD));
    }

    @Test
    void inPlaceValidationRejectsExpansionNarrowingAndOverlappingDestinations() {
        Tensor compact = new Tensor(DType.FLOAT16, 1, 2), expanded = TensorViews.expand(compact, 3, 2);
        assertThrows(IllegalStateException.class, () -> graph.layoutUnaryInto(expanded, ComputeGraph.OP_NEG, 0));
        assertThrows(IllegalStateException.class,
                () -> graph.layoutBinaryInto(expanded, expanded, ComputeGraph.OP_ADD));
        assertThrows(IllegalArgumentException.class,
                () -> graph.layoutBinaryInto(compact, new Tensor(1, 2), ComputeGraph.OP_ADD));
        assertThrows(IllegalArgumentException.class,
                () -> graph.layoutBinaryInto(compact, new Tensor(DType.FLOAT16, 3, 2), ComputeGraph.OP_ADD));
        assertTrue(graph.pendingAllocs.isEmpty());
    }
}
