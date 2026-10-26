package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;

class MetalRuntimeValidationTest {

    private final MetalGpuRuntime runtime = new MetalGpuRuntime();

    @Test
    void invalidAllocationsAreRejectedBeforeCallingNativeCode() {
        assertThrows(NullPointerException.class, () -> runtime.allocBuffers(null, new int[]{2}, 1));
        assertThrows(NullPointerException.class, () -> runtime.allocBuffers(new int[]{1}, null, 1));
        assertThrows(IllegalArgumentException.class, () -> runtime.allocBuffers(new int[0], new int[0], -1));
        assertThrows(IllegalArgumentException.class, () -> runtime.allocBuffers(new int[0], new int[]{2}, 1));
        assertThrows(IllegalArgumentException.class, () -> runtime.allocBuffers(new int[]{1}, new int[0], 1));
        assertThrows(IllegalArgumentException.class, () -> runtime.allocBuffers(new int[]{-1}, new int[]{2}, 1));
        assertThrows(IllegalArgumentException.class, () -> runtime.allocBuffers(new int[]{1}, new int[]{0}, 1));
    }

    @Test
    void invalidTransfersAreRejectedBeforeCallingNativeCode() {
        assertThrows(IllegalArgumentException.class, () -> runtime.uploadBuffer(-1, new float[1]));
        assertThrows(IllegalArgumentException.class, () -> runtime.downloadBuffer(-1, new float[1]));
        assertThrows(NullPointerException.class, () -> runtime.uploadBuffer(1, null));
        assertThrows(NullPointerException.class, () -> runtime.downloadBuffer(1, null));
        assertThrows(IllegalArgumentException.class, () -> runtime.zeroBuffer(-1, 1));
        assertThrows(IllegalArgumentException.class, () -> runtime.zeroBuffer(1, 0));
    }

    @Test
    void typedTransportRejectsInvalidRequestsWithoutCallingNativeCode() {
        assertThrows(NullPointerException.class,
                () -> runtime.allocTypedBuffers(new int[]{1}, new int[]{2}, null, 1));
        assertThrows(IllegalArgumentException.class,
                () -> runtime.allocTypedBuffers(new int[]{1}, new int[]{2}, new DType[0], 1));
        assertThrows(NullPointerException.class,
                () -> runtime.allocTypedBuffers(new int[]{1}, new int[]{2}, new DType[]{null}, 1));
        assertThrows(IllegalArgumentException.class, () -> runtime.uploadBuffer16(-1, new short[1]));
        assertThrows(IllegalArgumentException.class, () -> runtime.downloadBuffer16(-1, new short[1]));
        assertThrows(NullPointerException.class, () -> runtime.uploadBuffer16(1, null));
        assertThrows(NullPointerException.class, () -> runtime.downloadBuffer16(1, null));
        assertThrows(NullPointerException.class, () -> runtime.zeroBuffer(1, 1, null));
        assertThrows(IllegalArgumentException.class, () -> runtime.zeroBuffer(1, 0, DType.FLOAT16));
        assertThrows(IllegalArgumentException.class, () -> runtime.zeroBuffer(-1, 1, DType.FLOAT16));
    }

    @Test
    void releaseCountMustFitItsArray() {
        assertThrows(NullPointerException.class, () -> runtime.releaseBuffers(null, 0));
        assertThrows(IllegalArgumentException.class, () -> runtime.releaseBuffers(new int[0], -1));
        assertThrows(IllegalArgumentException.class, () -> runtime.releaseBuffers(new int[0], 1));
    }

    @Test
    void commandLengthMustFitItsArray() {
        assertThrows(NullPointerException.class, () -> runtime.flushOps(null, 0));
        assertThrows(IllegalArgumentException.class, () -> runtime.flushOps(new int[0], -1));
        assertThrows(IllegalArgumentException.class, () -> runtime.flushOps(new int[0], 1));
    }
}
