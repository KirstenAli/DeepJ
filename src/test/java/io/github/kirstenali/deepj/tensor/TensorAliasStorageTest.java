package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.autograd.GradMode;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TensorAliasStorageTest extends ComputeGraphTestSupport {
    @Test
    void matrixGraphCannotSilentlyTreatNdTypedOrStridedValuesAsDenseMatrices() {
        Tensor matrix = Tensor.ones(2, 3);
        Tensor strided = Tensor.aliasStorage(matrix, LayoutTransforms.permute(matrix.layout(), 1, 0));
        assertThrows(UnsupportedOperationException.class, () -> graph.ensureGpuBuffer(strided));
        assertThrows(UnsupportedOperationException.class, () -> graph.ensureGpuBuffer(new Tensor(2, 3, 4)));
        assertThrows(UnsupportedOperationException.class,
                () -> graph.copyToDevice(new Tensor(DType.FLOAT16, 2, 3)));
        assertTrue(graph.pendingAllocs.isEmpty());
    }

    @Test
    void bindingCannotCreateCompetingPhysicalOwnersOrMismatchedStorage() {
        GpuBuffer buffer = graph.newOutputBuffer(1, 2);
        assertThrows(IllegalArgumentException.class, () -> graph.bindTensorToBuffer(Tensor.ones(1, 1), buffer));
        Tensor source = graph.createOutputTensor(buffer);
        assertThrows(IllegalArgumentException.class, () -> graph.createOutputTensor(buffer));
        Tensor alias = Tensor.aliasStorage(source);
        graph.bindTensorToBuffer(alias, buffer);
        assertSame(buffer, source.getGpuTag());
        assertSame(source.memory(), graph.bufIdToMemory.get(buffer.id).get());
    }

    @Test
    void aliasesSharePhysicalDataAndVersionsButNotGradientMetadata() {
        Tensor source = Tensor.ones(1, 2).requiresGrad(true);
        Tensor alias = Tensor.aliasStorage(source);
        assertSame(source.memory(), alias.memory());
        assertNotSame(source.gradientState(), alias.gradientState());
        assertFalse(alias.requiresGrad());
        assertThrows(IllegalStateException.class, () -> alias.set(0, 0, 5));
        try (var ignored = GradMode.noGrad()) { alias.set(0, 0, 5); }
        assertEquals(5, source.get(0, 0));
        assertEquals(source.gradientState().version(), alias.gradientState().version());
        assertEquals(1, source.gradientState().version());
        assertNull(alias.grad());
    }

    @Test
    void aliasMutationInvalidatesActualSavedBackwardInputs() {
        Tensor source = Tensor.ones(1, 2).requiresGrad(true);
        Tensor output = source.multiply(source).sumTensor();
        Tensor alias = Tensor.aliasStorage(source);
        try (var ignored = GradMode.noGrad()) { alias.set(0, 1, 4); }
        assertThrows(IllegalStateException.class, output::backward);
    }

    @Test
    void allocationBindingDoesNotDependOnWhichLogicalTensorWasRegisteredLast() {
        Tensor source = graph.createOutputTensor(graph.newOutputBuffer(1, 2));
        Tensor alias = Tensor.aliasStorage(source);
        GpuBuffer buffer = graph.ensureGpuBuffer(alias);
        assertSame(source.memory(), graph.bufIdToMemory.get(buffer.id).get());
        assertSame(buffer, source.getGpuTag());
        assertSame(buffer, alias.getGpuTag());
        graph.flush();
        assertTrue(buffer.allocatedOnGpu);
        assertTrue(runtime.releaseCalls.isEmpty());
    }

    @Test
    void liveAliasPreventsTemporaryAndBackwardRetirement() {
        Tensor source = graph.createOutputTensor(graph.newOutputBuffer(1, 2));
        Tensor alias = Tensor.aliasStorage(source);
        GpuBuffer buffer = (GpuBuffer) source.getGpuTag();
        graph.retireBackwardBuffer(buffer);
        graph.releaseTemporary();
        assertSame(buffer, alias.getGpuTag());
        assertFalse(graph.backwardBufferIds.contains(buffer.id));
        assertTrue(runtime.releaseCalls.isEmpty());
    }

    @Test
    void releaseAllMaterializesOneAllocationAndClearsEveryAlias() {
        Tensor source = graph.createOutputTensor(graph.newOutputBuffer(1, 2));
        Tensor alias = Tensor.aliasStorage(source);
        GpuBuffer buffer = (GpuBuffer) source.getGpuTag();
        runtime.downloadResult = new float[]{3, 7};
        graph.releaseAll();
        assertEquals(1, runtime.downloads.size());
        assertArrayEquals(new float[]{3, 7}, alias.rawData());
        assertSame(source.rawData(), alias.rawData());
        assertNull(source.getGpuTag());
        assertNull(alias.getGpuTag());
        assertThrows(IllegalStateException.class, () -> graph.createOutputTensor(buffer));
    }

    @Test
    void deviceRetentionIsBalancedPerAliasAndSharedByAllocation() {
        Tensor source = Tensor.ones(1, 2).retainDeviceBuffer();
        Tensor alias = Tensor.aliasStorage(source).retainDeviceBuffer();
        source.releaseDeviceBufferRetention();
        source.releaseDeviceBufferRetention();
        assertTrue(source.retainsDeviceBuffer());
        alias.releaseDeviceBufferRetention();
        assertFalse(source.retainsDeviceBuffer());
        assertFalse(alias.retainsDeviceBuffer());
    }

    @Test
    void hostMutationThroughAliasReuploadsSharedAllocation() {
        Tensor source = Tensor.ones(1, 2);
        GpuBuffer buffer = graph.ensureGpuBuffer(source);
        Tensor alias = Tensor.aliasStorage(source);
        graph.flush();
        alias.set(0, 0, 8);
        assertTrue(buffer.needsUpload);
        assertSame(buffer, graph.ensureGpuBuffer(source));
        graph.recordUnary(ComputeGraph.OP_NEG, buffer, graph.newOutputBuffer(1, 2));
        graph.flush();
        assertArrayEquals(new float[]{8, 1}, runtime.uploads.get(1).data());
    }

    @Test
    void movingAliasToAnotherGraphPreservesDataAndSharedBinding() {
        Tensor source = graph.createOutputTensor(graph.newOutputBuffer(1, 2));
        Tensor alias = Tensor.aliasStorage(source);
        GpuBuffer old = (GpuBuffer) source.getGpuTag();
        var other = new ComputeGraph(runtime);
        runtime.downloadResult = new float[]{5, 9};
        GpuBuffer moved = other.ensureGpuBuffer(alias);
        assertNotSame(old, moved);
        assertSame(moved, source.getGpuTag());
        assertArrayEquals(new float[]{5, 9}, other.pendingUploadData.get(0));
        graph.flush();
        assertThrows(IllegalStateException.class, old::requireLive);
        assertSame(moved, alias.getGpuTag());
        other.releaseAll();
    }
}
