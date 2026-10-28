package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;
import java.util.Arrays;
import static org.junit.jupiter.api.Assertions.*;

class OutputShapeOwnershipTest {
    @Test
    void outputLikeSharesOnlyImmutableDimensionsAcrossEveryStorageFormat() {
        var graph = new ComputeGraph(new GpuStorageRuntime());
        Tensor source = Tensor.ones(DType.BFLOAT16, 2, 3).transpose(0, 1);
        for (DType dtype : DType.values()) {
            GpuBuffer buffer = graph.newOutputBuffer(1, 6, dtype);
            Tensor output = graph.createOutputLike(buffer, source);
            assertSame(source.layout().shape(), output.layout().shape());
            assertNotSame(source.memory(), output.memory());
            assertNotSame(source.gradientState(), output.gradientState());
            assertTrue(output.isContiguous());
            assertFalse(output.memory().hasHostStorage());
            assertEquals(dtype, output.dtype());
            assertArrayEquals(new int[]{3, 2}, output.shape());
        }
        graph.releaseAll();
    }

    @Test
    void outputLikeRejectsInvalidElementCountBeforeBindingStorage() {
        var graph = new ComputeGraph(new GpuStorageRuntime());
        Tensor source = Tensor.ones(DType.FLOAT16, 2, 3);
        GpuBuffer buffer = graph.newOutputBuffer(1, 4);
        assertThrows(IllegalArgumentException.class, () -> graph.createOutputLike(buffer, source));
        assertFalse(graph.bufIdToMemory.containsKey(buffer.id));
        assertNull(source.getGpuTag());
        assertThrows(IllegalArgumentException.class, () -> graph.createOutputLike(buffer, new Tensor(0)));
        graph.releaseAll();
    }

    @Test
    void internalDeviceFactorySharesOnlyImmutableShapeMetadataNotCallerArrays() {
        for (DType dtype : DType.values()) {
            int[] dimensions = {2, 3};
            var shape = new TensorShape(dimensions);
            Tensor tensor = Tensor.deviceOnly(dtype, TensorLayout.dense(shape));
            dimensions[0] = 99;
            assertSame(shape, tensor.layout().shape());
            assertArrayEquals(new int[]{2, 3}, tensor.shape());
            assertEquals(6L * dtype.bytes(), tensor.storageBytes());
            assertFalse(tensor.memory().hasHostStorage());
        }
    }

    @Test
    void graphOutputRanksAndDtypesKeepIndependentCallerShapeSnapshots() {
        var graph = new ComputeGraph(new GpuStorageRuntime());
        for (DType dtype : DType.values()) {
            for (int rank = 0; rank <= 6; rank++) { check(graph, dtype, rank); }
        }
        graph.releaseAll();
    }

    @Test
    void canonicalOutputGeometryIsSharedWithoutStorageGradientsOrMutationVersions() {
        var graph = new ComputeGraph(new GpuStorageRuntime());
        Tensor source = Tensor.ones(DType.FLOAT32, 2, 3).requiresGrad(true);
        Tensor output = graph.createOutputLike(graph.newOutputBuffer(2, 3), source);
        assertSame(source.layout(), output.layout());
        assertNotSame(source.memory(), output.memory());
        assertNotSame(source.gradientState(), output.gradientState());
        long version = output.gradientState().version();
        assertThrows(IllegalStateException.class, () -> source.set(0, 0, 7));
        try (var ignored = io.github.kirstenali.deepj.autograd.GradMode.noGrad()) {
            source.set(0, 0, 7);
        }
        assertEquals(version, output.gradientState().version());
        long[] strides = output.strides();
        strides[0] = 99;
        assertArrayEquals(new long[]{3, 1}, source.strides());
        graph.releaseAll();
    }

    @Test
    void singletonAndOffsetViewsStillProduceCanonicalIndependentOutputGeometry() {
        var graph = new ComputeGraph(new GpuStorageRuntime());
        Tensor singleton = Tensor.ones(DType.FLOAT32, 2, 1).transpose(0, 1);
        Tensor offset = Tensor.ones(DType.FLOAT32, 3, 2).slice(0, 1, 3);
        Tensor prefix = Tensor.ones(DType.FLOAT32, 3, 2).slice(0, 0, 2);
        for (Tensor source : new Tensor[]{singleton, offset, prefix}) {
            Tensor output = graph.createOutputLike(graph.newOutputBuffer(1, source.numel()), source);
            assertNotSame(source.layout(), output.layout());
            assertArrayEquals(source.shape(), output.shape());
            assertArrayEquals(new TensorShape(source.shape()).denseStrides(), output.strides());
            assertEquals(0, output.storageOffset());
        }
        graph.releaseAll();
    }

    private static void check(ComputeGraph graph, DType dtype, int rank) {
        int[] shape = new int[rank];
        Arrays.fill(shape, 2);
        GpuBuffer buffer = graph.newOutputBuffer(1, 1 << rank, dtype);
        Tensor output = graph.createOutputTensor(buffer, shape);
        Arrays.fill(shape, 99);
        assertEquals(rank, output.rank());
        assertEquals(1 << rank, output.numel());
        assertEquals(dtype, output.dtype());
        assertSame(buffer, output.getGpuTag());
        assertFalse(output.memory().hasHostStorage());
        for (int axis = 0; axis < rank; axis++) { assertEquals(2, output.size(axis)); }
    }
}
