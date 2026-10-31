package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class NdStatisticsTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void meansAndVariancesFollowExplicitAxesAndCorrection() {
        Tensor input = Tensor.from(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 2, 2, 3);
        assertArrayEquals(new float[]{4, 5, 6, 7, 8, 9}, input.mean(0, false).toFloatArray());
        assertArrayEquals(new float[]{2, 5, 8, 11}, input.mean(-1, true).toFloatArray());
        assertArrayEquals(new int[]{2, 2, 1}, input.mean(-1, true).shape());
        assertEquals(6.5f, input.meanAxes().item());
        assertArrayEquals(new float[]{2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f},
                input.variance(1, false).toFloatArray());
        assertArrayEquals(new float[]{4.5f, 4.5f, 4.5f, 4.5f, 4.5f, 4.5f},
                input.variance(1, false, 1).toFloatArray());
    }

    @Test
    void compactStatisticsDecodeInputsAndRoundOnlyTheCompletedOutput() {
        Tensor values = Tensor.from(new float[]{2048, 1, 1}, DType.FLOAT16);
        float expected = Float.float16ToFloat(Float.floatToFloat16(2050 / 3.0f));
        assertEquals(expected, values.meanAxes().item());
        Tensor matrix = Tensor.from(new float[]{1, 3, 3, 5}, DType.BFLOAT16, 2, 2);
        assertArrayEquals(new float[]{1, 1}, matrix.varianceAlongRows().toFloatArray());
        assertArrayEquals(new float[]{3, 5}, matrix.maxAlongRows().toFloatArray());
        assertEquals(DType.BFLOAT16, matrix.mean(1, false).dtype());
    }

    @Test
    void maximaPropagateNanAndRetainSignedZeroSelectionSemantics() {
        Tensor input = Tensor.from(new float[]{-0.0f, 0.0f, -2, -3}, 2, 2).transpose(0, 1);
        assertArrayEquals(new float[]{0, -2}, input.maximum(0, false).toFloatArray());
        Tensor nan = Tensor.from(new float[]{1, Float.NaN, 2});
        assertTrue(Float.isNaN(nan.maxAxes().item()));
        assertEquals(0, new Tensor(0, 3).maximum(1, false).numel());
        assertThrows(IllegalArgumentException.class, () -> new Tensor(0, 3).maximum(0, false));
        assertEquals(7, Tensor.from(new float[]{7}, new int[0]).maxAxes().item());
    }

    @Test
    void emptyStatisticsAndInvalidReductionsAreExplicit() {
        Tensor empty = new Tensor(0, 3);
        assertTrue(Float.isNaN(empty.meanAxes().item()));
        assertTrue(Float.isNaN(empty.varianceAxes(new int[0], false, 0).item()));
        assertTrue(Float.isNaN(Tensor.from(new float[]{1}).variance(0, false, 1).item()));
        assertThrows(IllegalArgumentException.class, () -> empty.variance(0, false, -1));
        assertThrows(IllegalArgumentException.class, () -> empty.meanAxes(0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> empty.maximum(2, false));
        assertThrows(IllegalArgumentException.class, () -> Tensor.backend().sumRows(new Tensor(2, 3, 4)));
    }

    @Test
    void malformedMaximumGradientsAreRejectedBeforeArithmetic() {
        Tensor input = Tensor.from(new float[]{1, 2, 3, 4}, 2, 2);
        Tensor maximum = input.maximum(1, false);
        assertThrows(IllegalArgumentException.class, () -> Tensor.backend().maximumAxesBackward(
                input, new Tensor(2, 1), new Tensor(2, 1), new int[]{1}, false));
        assertThrows(IllegalArgumentException.class, () -> Tensor.backend().maximumAxesBackward(
                input, maximum, new Tensor(1), new int[]{1}, false));
    }
}
