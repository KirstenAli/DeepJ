package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class TensorHostReadTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest @EnumSource(DType.class)
    void snapshotsRespectContiguousOffsetsPaddedRowsAndStrides(DType dtype) {
        Tensor source = Tensor.from(new float[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, dtype, 3, 4);
        check(source);
        check(source.slice(0, 1, 3));
        check(source.slice(1, 1, 3));
        check(source.transpose(0, 1));
        check(source.slice(1, 0, 4, 2));
        check(source.slice(0, 0, 1).expand(3, 4));
        check(source.slice(0, 3, 3));
        check(source.slice(1, 4, 4));
    }

    @ParameterizedTest @EnumSource(DType.class)
    void snapshotsPreserveSpecialValuesAndNeverAliasStorage(DType dtype) {
        Tensor source = Tensor.from(new float[]{-0.0f, Float.NaN, Float.POSITIVE_INFINITY,
                Float.NEGATIVE_INFINITY, Float.MIN_VALUE, -0.00006103515625f}, dtype, 2, 3);
        check(source);
        float[] flat = source.toFloatArray(), row = source.rowData(0);
        flat[0] = 9;
        row[1] = 10;
        assertEquals(Float.floatToRawIntBits(-0.0f), Float.floatToRawIntBits(source.get(0, 0)));
        assertTrue(Float.isNaN(source.get(0, 1)));
    }

    private static void check(Tensor tensor) {
        float[] flat = tensor.toFloatArray();
        for (int row = 0; row < tensor.matrixRows(); row++) {
            float[] values = tensor.rowData(row);
            for (int column = 0; column < values.length; column++) {
                int expected = Float.floatToRawIntBits(tensor.get(row, column));
                assertEquals(expected, Float.floatToRawIntBits(values[column]));
                assertEquals(expected, Float.floatToRawIntBits(flat[row * values.length + column]));
            }
        }
    }
}
