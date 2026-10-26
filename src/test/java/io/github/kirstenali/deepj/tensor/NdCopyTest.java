package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class NdCopyTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void float32CopyPreservesSignedZeroAndNanPayloadsInLogicalOrder() {
        Tensor input = Tensor.from(new float[]{-0.0f, Float.intBitsToFloat(0x7fc12345), 2, 3}, 2, 2)
                .transpose(0, 1);
        Tensor copy = input.copy();
        assertEquals(0x80000000, Float.floatToRawIntBits(copy.get(0, 0)));
        assertEquals(0x7fc12345, Float.floatToRawIntBits(copy.get(1, 0)));
        assertNotSame(input.memory(), copy.memory());
        assertTrue(copy.isContiguous());
    }

    @Test
    void compactCopiesPreserveRawBitsRatherThanDecodeAndReencode() {
        for (DType type : new DType[]{DType.FLOAT16, DType.BFLOAT16}) {
            Tensor input = new Tensor(type, 2, 2);
            short[] bits = input.memory().shorts();
            bits[0] = (short) 0x8000;
            bits[1] = (short) 0x7f81;
            bits[2] = 1;
            bits[3] = (short) 0xffff;
            Tensor copy = input.transpose(0, 1).copy();
            assertArrayEquals(new short[]{bits[0], bits[2], bits[1], bits[3]}, copy.memory().shorts());
            assertEquals(8, copy.storageBytes());
        }
    }

    @Test
    void copyingTracksIdentityWhileDetachedStorageDoesNotAliasOrTrack() {
        Tensor input = Tensor.from(new float[]{1, 2}, DType.FLOAT16).requiresGrad(true);
        Tensor copy = input.copy(), detached = input.detach();
        assertFalse(detached.requiresGrad());
        detached.setAt(7, 0);
        assertEquals(1, input.get(0));
        copy.sumTensor().backward();
        assertArrayEquals(new float[]{1, 1}, input.grad().toFloatArray());
        assertEquals(DType.FLOAT32, input.grad().dtype());
        try (var ignored = GradMode.noGrad()) { input.addScalarInPlace(1); }
        assertArrayEquals(new float[]{1, 2}, copy.toFloatArray());
    }

    @Test
    void identityCopiesStillEnforceTheExistingGraphInputMutationGuard() {
        Tensor input = Tensor.from(new float[]{1, 2}).requiresGrad(true), output = input.copy();
        try (var ignored = GradMode.noGrad()) { input.addScalarInPlace(1); }
        assertThrows(IllegalStateException.class, () -> output.sumTensor().backward());
    }
}
