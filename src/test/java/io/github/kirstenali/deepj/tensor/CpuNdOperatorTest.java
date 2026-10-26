package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CpuNdOperatorTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void trailingBroadcastsScalarsAndCompactDtypesUseLogicalValues() {
        for (DType dtype : DType.values()) {
            Tensor input = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, dtype, 2, 1, 3);
            Tensor bias = Tensor.from(new float[]{1, 2, 4}, dtype);
            assertArrayEquals(new float[]{2, 4, 7, 5, 7, 10}, input.add(bias).toFloatArray());
            assertArrayEquals(new float[]{1, 4, 12, 4, 10, 24}, input.multiply(bias).toFloatArray());
            assertEquals(dtype, input.add(bias).dtype());
            Tensor scalar = Tensor.from(new float[]{2}, dtype, new int[0]);
            assertArrayEquals(new float[]{3, 4, 5, 6, 7, 8}, input.add(scalar).toFloatArray());
        }
    }

    @Test
    void stridedAndOffsetInputsHaveNoFlatBackingAssumptions() {
        Tensor input = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, 2, 3).permute(1, 0);
        assertArrayEquals(new float[]{2, 8, 4, 10, 6, 12}, input.multiplyScalar(2).toFloatArray());
        assertArrayEquals(new float[]{2, 5, 3, 6, 4, 7}, input.addScalar(1).toFloatArray());
        assertArrayEquals(new float[]{4, 5, 6}, input.select(1, 1).toFloatArray());
        assertArrayEquals(new float[]{1, 4, 2, 5, 3, 6}, input.reshape(6).toFloatArray());
        assertThrows(IllegalArgumentException.class, () -> input.view(6));
        assertTrue(input.contiguous().isContiguous());
    }

    @Test
    void overlappingInputAliasesAreSnapshottedBeforeInPlaceWrites() {
        Tensor input = Tensor.from2D(new float[][]{{1, 2}, {3, 4}});
        input.addInPlace(input.transpose(0, 1));
        assertArrayEquals(new float[]{2, 5, 5, 8}, input.toFloatArray());
        Tensor rows = Tensor.from2D(new float[][]{{1, 2}, {3, 4}});
        rows.addInPlace(rows.select(0, 0));
        assertArrayEquals(new float[]{2, 4, 4, 6}, rows.toFloatArray());
        Tensor expanded = Tensor.from(new float[]{1}, 1).expand(4);
        assertThrows(IllegalStateException.class, () -> expanded.addScalarInPlace(1));
    }

    @Test
    void incompatibleBroadcastsExpansionAndImplicitNarrowingAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> new Tensor(2, 3).add(new Tensor(4)));
        Tensor narrow = new Tensor(1, 3);
        assertThrows(IllegalArgumentException.class, () -> narrow.addInPlace(new Tensor(2, 3)));
        assertEquals(0, narrow.gradientState().version());
        Tensor compact = new Tensor(DType.FLOAT16, 3);
        assertThrows(IllegalArgumentException.class, () -> compact.addInPlace(new Tensor(3)));
        assertEquals(0, compact.gradientState().version());
        Tensor mixed = compact.add(new Tensor(DType.BFLOAT16, 3));
        assertEquals(DType.FLOAT32, mixed.dtype());
        assertEquals(12, mixed.storageBytes());
    }

    @Test
    void axisReductionsPreserveOrRemoveDimensionsIncludingEmptyAndScalarCases() {
        Tensor input = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, 2, 1, 3);
        assertArrayEquals(new float[]{5, 7, 9}, input.sum(0, false).toFloatArray());
        assertArrayEquals(new int[]{2, 1, 1}, input.sum(-1, true).shape());
        assertArrayEquals(new float[]{6, 15}, input.sumAxes(1, 2).toFloatArray());
        assertEquals(21, input.sumTensor().item());
        assertEquals(0, new Tensor(2, 0, 3).sumTensor().item());
        assertArrayEquals(new float[]{0, 0, 0}, new Tensor(0, 3).sum(0, false).toFloatArray());
        assertEquals(4, Tensor.from(new float[]{4}, new int[0]).sumTensor().item());
        assertThrows(IllegalArgumentException.class, () -> input.sumAxes(0, -3));
        assertThrows(IndexOutOfBoundsException.class, () -> input.sum(3, false));
    }

    @Test
    void stableSoftmaxWorksAcrossNonLastAxesAndEmptyInputs() {
        Tensor input = Tensor.from(new float[]{1000, 1000, 1000, 1000}, 2, 1, 2).permute(2, 1, 0);
        assertArrayEquals(new float[]{0.5f, 0.5f, 0.5f, 0.5f}, input.softmax(0).toFloatArray());
        Tensor compact = input.to(DType.BFLOAT16).softmax(-1);
        assertEquals(DType.BFLOAT16, compact.dtype());
        assertArrayEquals(new float[]{0.5f, 0.5f, 0.5f, 0.5f}, compact.toFloatArray());
        assertEquals(0, new Tensor(2, 0, 3).softmax(1).numel());
        assertThrows(IndexOutOfBoundsException.class, () -> input.softmax(3));
    }

    @Test
    void slicesNormalizeNegativeBoundsAndKeepEmptyViewsValid() {
        Tensor input = Tensor.from(new float[]{0, 1, 2, 3, 4, 5});
        assertArrayEquals(new float[]{3, 5}, input.slice(0, -3, 99, 2).toFloatArray());
        assertEquals(0, input.slice(0, 5, 2).numel());
        assertEquals(0, input.slice(0, 90, 99).numel());
        assertThrows(IllegalArgumentException.class, () -> input.slice(0, 0, 3, 0));
        assertThrows(IllegalArgumentException.class, () -> input.permute(0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> input.transpose(0, 1));
    }

    @Test
    void compactReductionsRoundOnlyTheCompletedFloat32Sum() {
        Tensor input = Tensor.from(new float[]{2048, 1, 1}, DType.FLOAT16);
        assertEquals(2050, input.sumTensor().item());
        Tensor same = input.to(DType.FLOAT16);
        assertSame(input, same);
        Tensor widened = input.to(DType.FLOAT32);
        assertEquals(DType.FLOAT32, widened.dtype());
        assertArrayEquals(new float[]{2048, 1, 1}, widened.toFloatArray());
        assertSame(widened, widened.contiguous());
    }
}
