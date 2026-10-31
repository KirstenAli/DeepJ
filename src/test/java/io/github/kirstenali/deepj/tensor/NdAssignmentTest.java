package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class NdAssignmentTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void assignmentExplicitlyConvertsStorageAndPreservesShape() {
        Tensor input = Tensor.from(new float[]{0.1f, 1.3f, -0.7f, 2.1f}, 2, 1, 2);
        for (DType type : DType.values()) {
            Tensor target = new Tensor(type, input.shape());
            assertSame(target, target.copyFrom(input));
            assertArrayEquals(input.to(type).toFloatArray(), target.toFloatArray());
            assertEquals((long) input.numel() * type.bytes(), target.storageBytes());
        }
    }

    @Test
    void overlappingCopiesSnapshotBeforeWritingAndPreserveBits() {
        Tensor input = Tensor.from(new float[]{1, 2, 3, 4}, 2, 2);
        input.copyFrom(input.transpose(0, 1));
        assertArrayEquals(new float[]{1, 3, 2, 4}, input.toFloatArray());
        Tensor source = Tensor.from(new float[]{-0.0f, Float.intBitsToFloat(0x7fc12345)}, 2, 1);
        input.slice(1, 0, 1).copyFrom(source);
        assertEquals(0x80000000, Float.floatToRawIntBits(input.get(0, 0)));
        assertEquals(0x7fc12345, Float.floatToRawIntBits(input.get(1, 0)));
        assertSame(input, input.copyFrom(input));
    }

    @Test
    void offsetAndStridedMixedCopiesUseLogicalRatherThanAllocationIndices() {
        Tensor input = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, 3, 2);
        Tensor target = new Tensor(DType.BFLOAT16, 3, 2);
        target.transpose(0, 1).copyFrom(input.transpose(0, 1));
        assertArrayEquals(input.toFloatArray(), target.toFloatArray());
        target.slice(0, 1, 3).copyFrom(input.slice(0, 0, 2));
        assertArrayEquals(new float[]{1, 2, 1, 2, 3, 4}, target.toFloatArray());
    }

    @Test
    void invalidAssignmentCannotMutateOrBypassGradientGuards() {
        Tensor input = Tensor.from(new float[]{1}).requiresGrad(true), target = new Tensor(1);
        assertThrows(IllegalStateException.class, () -> target.copyFrom(input));
        assertThrows(IllegalArgumentException.class, () -> target.copyFrom(new Tensor(2)));
        assertEquals(0, target.gradientState().version());
        assertThrows(IllegalStateException.class, () -> target.expand(2).copyFrom(new Tensor(2)));
        try (var ignored = GradMode.noGrad()) { target.copyFrom(input); }
        assertEquals(1, target.item());
        assertFalse(target.requiresGrad());
    }
}
