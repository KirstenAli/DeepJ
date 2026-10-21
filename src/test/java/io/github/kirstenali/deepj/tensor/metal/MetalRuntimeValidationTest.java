package io.github.kirstenali.deepj.tensor.metal;

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
