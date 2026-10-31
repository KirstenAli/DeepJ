package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MetalNative16CapabilityTest {
    @Test
    void float32NeverSelectsNative16AndNullIsRejectedBeforeNativeAccess() {
        assertFalse(MetalBackend.supportsNative16Matmul(DType.FLOAT32));
        assertThrows(NullPointerException.class, () -> MetalBackend.supportsNative16Matmul(null));
    }
}
