package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorTestStorage;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.*;

class MetalNdAssignmentTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(DType.class)
    void stridedAssignmentsPreserveParentElementsAndSharedStorageIdentity(DType dtype) {
        Tensor parent = metal.addScalar(metal.zeros(3, 4), 1).to(dtype);
        Tensor target = parent.slice(0, 1, 3, 1).slice(1, 0, 4, 2);
        Tensor source = Tensor.from(new float[]{2.25f, 3.5f, 4.75f, 6}, 2, 2);
        Object allocation = parent.getGpuTag();
        target.copyFrom(source);
        assertSame(allocation, parent.getGpuTag());
        assertSame(allocation, target.getGpuTag());
        assertFalse(TensorTestStorage.hasHostStorage(parent));
        assertArrayEquals(new float[]{1, 1, 1, 1, 2.25f, 1, 3.5f, 1, 4.75f, 1, 6, 1}, parent.toFloatArray());
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void aliasedAssignmentsReadASnapshotInsteadOfRacingGpuWrites(DType dtype) {
        Tensor square = Tensor.from(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, dtype, 3, 3);
        square.copyFrom(square.permute(1, 0));
        assertArrayEquals(new float[]{1, 4, 7, 2, 5, 8, 3, 6, 9}, square.toFloatArray());
        Tensor shifted = Tensor.from(new float[]{0, 1, 2, 3, 4, 5}, dtype);
        shifted.slice(0, 1, 6, 1).copyFrom(shifted.slice(0, 0, 5, 1));
        assertArrayEquals(new float[]{0, 0, 1, 2, 3, 4}, shifted.toFloatArray());
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void stridedSameFormatAssignmentPreservesEveryRaw16BitPayload(DType dtype) {
        short[] bits = new short[65_536];
        for (int index = 0; index < bits.length; index++) { bits[index] = (short) index; }
        Tensor source = TensorTestStorage.from16Bits(bits, dtype, 256, 256).permute(1, 0);
        Tensor parent = new Tensor(dtype, 256, 512), target = parent.slice(1, 0, 512, 2);
        target.copyFrom(source);
        short[] result = TensorTestStorage.bits16(parent);
        for (int index = 0; index < bits.length; index++) {
            assertEquals(bits[(index % 256) * 256 + index / 256], result[index * 2]);
            assertEquals(0, result[index * 2 + 1]);
        }
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void inPlaceBinaryAndScalarOperationsUpdateOnlyTheView(DType dtype) {
        Tensor parent = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, dtype, 2, 3);
        Tensor target = parent.permute(1, 0).slice(0, 0, 3, 2);
        Tensor factor = Tensor.from(new float[]{2, 4}, dtype);
        target.addInPlace(factor).subtractInPlace(factor).multiplyInPlace(factor).divideInPlace(factor);
        target.addScalarInPlace(2).multiplyScalarInPlace(2).divideScalarInPlace(2);
        assertArrayEquals(new float[]{3, 2, 5, 6, 5, 8}, parent.toFloatArray());
        target.negInPlace().negInPlace().sqrtInPlace();
        float[] expected = cpu.sqrt(Tensor.from(new float[]{3, 6, 5, 8}, dtype, 2, 2)).toFloatArray();
        assertArrayEquals(expected, target.toFloatArray(), dtype == DType.FLOAT32 ? 3e-6f : 0);
    }

    @Test
    void invalidWritesAndMutationsCannotBypassShapeDtypeOrSavedInputGuards() {
        Tensor compact = new Tensor(DType.FLOAT16, 1, 2), broadcast = compact.expand(3, 2);
        assertThrows(IllegalStateException.class, () -> broadcast.addScalarInPlace(1));
        assertThrows(IllegalStateException.class, () -> metal.copyInto(broadcast, new Tensor(3, 2)));
        assertThrows(IllegalArgumentException.class, () -> metal.addInPlace(compact, new Tensor(1, 2)));
        assertThrows(IllegalArgumentException.class, () -> metal.addInPlace(compact, new Tensor(DType.FLOAT16, 3, 2)));
        assertThrows(IllegalArgumentException.class, () -> metal.copyInto(compact, new Tensor(2)));
        Tensor tracked = Tensor.from(new float[]{1, 2}, 1, 1, 2).requiresGrad(true);
        Tensor saved = tracked.multiply(tracked);
        assertThrows(IllegalStateException.class, () -> tracked.addScalarInPlace(1));
        saved.backward(Tensor.from(new float[]{1, 1}, 1, 1, 2));
        assertArrayEquals(new float[]{2, 4}, tracked.grad().toFloatArray());
    }
}
