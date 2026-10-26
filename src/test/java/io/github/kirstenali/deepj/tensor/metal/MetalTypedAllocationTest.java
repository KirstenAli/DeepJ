package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.ComputeGraph;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MetalTypedAllocationTest extends MetalDifferentialTestSupport {
    @Test
    void typedAllocationArraysAreValidatedBeforeAnyDeviceAllocation() {
        invalid(null, new int[]{2}, new int[]{2}, 1);
        invalid(new int[]{1}, null, new int[]{2}, 1);
        invalid(new int[]{1}, new int[]{2}, null, 1);
        invalid(new int[0], new int[]{2}, new int[]{2}, 1);
        invalid(new int[]{1}, new int[0], new int[]{2}, 1);
        invalid(new int[]{1}, new int[]{2}, new int[0], 1);
        invalid(new int[0], new int[0], new int[0], -1);
        invalid(new int[]{-1}, new int[]{2}, new int[]{2}, 1);
        invalid(new int[]{1}, new int[]{0}, new int[]{2}, 1);
        invalid(new int[]{1}, new int[]{2}, new int[]{4}, 1);
        MetalNative.nativeAllocTypedBuffers(new int[0], new int[0], new int[0], 0);
        assertThrows(RuntimeException.class, () -> MetalNative.nativeBufferByteLength(-1));
    }

    @Test
    void floatStorageCannotBeReadAsCompactEvenIfTheByteLengthWouldFit() {
        Tensor owner = new Tensor(1, 2);
        int id = metal.graph.ensureGpuBuffer(owner).id;
        metal.graph.flush();
        assertEquals(8, MetalNative.nativeBufferByteLength(id));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeUploadBuffer16(id, new short[4]));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeDownloadBuffer16(id, new short[4]));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeZeroTypedBuffer(id, 4, 2));
        invalid(new int[]{id}, new int[]{4}, new int[]{2}, 1);
        invalid(new int[]{id}, new int[]{2}, new int[]{0}, 1);
    }

    @Test
    void compactStorageCannotBeSentToAMatrixKernelThroughAnUncheckedCommandStream() {
        Tensor owner = new Tensor(DType.FLOAT16, 1, 2);
        int id = metal.graph.ensureStorageBuffer(owner).id;
        metal.graph.flush();
        int[] commands = {ComputeGraph.OP_NEG, id, id, 2};
        assertThrows(RuntimeException.class, () -> MetalNative.nativeFlushOps(commands, commands.length));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeZeroTypedBuffer(id, 1, 2));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeZeroTypedBuffer(id, 0, 2));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeZeroTypedBuffer(-1, 2, 2));
    }

    private static void invalid(int[] ids, int[] sizes, int[] types, int count) {
        assertThrows(RuntimeException.class, () -> MetalNative.nativeAllocTypedBuffers(ids, sizes, types, count));
    }
}
