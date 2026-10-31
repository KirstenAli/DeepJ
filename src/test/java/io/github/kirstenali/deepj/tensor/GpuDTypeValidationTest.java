package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class GpuDTypeValidationTest extends ComputeGraphTestSupport {
    @Test
    void dtypeCodesAreStableAndDoNotDependOnEnumOrdinal() {
        assertEquals(1, DType.FLOAT32.storageCode());
        assertEquals(2, DType.FLOAT16.storageCode());
        assertEquals(3, DType.BFLOAT16.storageCode());
        for (DType dtype : DType.values()) { assertSame(dtype, DType.fromStorageCode(dtype.storageCode())); }
        assertThrows(IllegalArgumentException.class, () -> DType.fromStorageCode(0));
        assertThrows(IllegalArgumentException.class, () -> DType.fromStorageCode(4));
    }

    @Test
    void bufferDescriptorsRejectInvalidDimensionsAndKeepCountsAndBytesDistinct() {
        assertThrows(IllegalArgumentException.class, () -> new GpuBuffer(-1, 1, 1, true));
        assertThrows(IllegalArgumentException.class, () -> new GpuBuffer(0, 0, 1, true));
        assertThrows(IllegalArgumentException.class, () -> new GpuBuffer(0, Integer.MAX_VALUE, 2, true));
        assertThrows(NullPointerException.class, () -> new GpuBuffer(0, 1, 1, null, true));
        GpuBuffer buffer = new GpuBuffer(0, 1, Integer.MAX_VALUE, DType.BFLOAT16, false);
        assertEquals(2L * Integer.MAX_VALUE, buffer.byteLength());
        assertThrows(UnsupportedOperationException.class, buffer::floatCount);
    }

    @Test
    void customFloatRuntimesDoNotSilentlyEmulateCompactStorage() {
        runtime.allocTypedBuffers(new int[]{1}, new int[]{2}, new DType[]{DType.FLOAT32}, 1);
        assertEquals(1, runtime.allocCalls.size());
        assertThrows(UnsupportedOperationException.class,
                () -> runtime.allocTypedBuffers(new int[]{2}, new int[]{3}, new DType[]{DType.FLOAT16}, 1));
        assertThrows(UnsupportedOperationException.class, () -> runtime.uploadBuffer16(1, new short[2]));
        assertThrows(UnsupportedOperationException.class, () -> runtime.downloadBuffer16(1, new short[2]));
        assertThrows(UnsupportedOperationException.class, () -> runtime.zeroBuffer(1, 2, DType.FLOAT16));
        assertEquals(1, runtime.allocCalls.size());
    }

    @Test
    void typedAllocationPreflightRejectsBadRequestsBeforeCallingTheRuntime() {
        assertThrows(NullPointerException.class, () -> runtime.allocTypedBuffers(null, new int[0], new DType[0], 0));
        assertThrows(NullPointerException.class, () -> runtime.allocTypedBuffers(new int[0], null, new DType[0], 0));
        assertThrows(NullPointerException.class, () -> runtime.allocTypedBuffers(new int[0], new int[0], null, 0));
        invalid(new int[0], new int[0], new DType[0], -1);
        invalid(new int[0], new int[]{1}, new DType[]{DType.FLOAT32}, 1);
        invalid(new int[]{1}, new int[0], new DType[]{DType.FLOAT32}, 1);
        invalid(new int[]{1}, new int[]{1}, new DType[0], 1);
        invalid(new int[]{-1}, new int[]{1}, new DType[]{DType.FLOAT32}, 1);
        invalid(new int[]{1}, new int[]{0}, new DType[]{DType.FLOAT32}, 1);
        assertThrows(NullPointerException.class,
                () -> runtime.allocTypedBuffers(new int[]{1}, new int[]{1}, new DType[]{null}, 1));
        assertTrue(runtime.allocCalls.isEmpty());
    }

    @Test
    void zeroMetadataAndEmptyStorageFailClearly() {
        assertThrows(IllegalArgumentException.class, () -> runtime.zeroBuffer(-1, 1, DType.FLOAT16));
        assertThrows(IllegalArgumentException.class, () -> runtime.zeroBuffer(1, 0, DType.FLOAT16));
        assertThrows(NullPointerException.class, () -> runtime.zeroBuffer(1, 1, null));
        assertThrows(IllegalArgumentException.class, () -> graph.ensureStorageBuffer(new Tensor(0, 3)));
        runtime.zeroBuffer(3, 2, DType.FLOAT32);
        assertArrayEquals(new float[2], runtime.uploads.get(0).data());
        assertTrue(graph.pendingAllocs.isEmpty());
    }

    private void invalid(int[] ids, int[] sizes, DType[] types, int count) {
        assertThrows(IllegalArgumentException.class, () -> runtime.allocTypedBuffers(ids, sizes, types, count));
    }
}
