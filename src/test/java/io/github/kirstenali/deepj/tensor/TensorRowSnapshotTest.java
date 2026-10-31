package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.io.*;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class TensorRowSnapshotTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest
    @EnumSource(DType.class)
    void rowSnapshotsRespectDtypeOffsetsStridesAndIndependentOwnership(DType dtype) {
        Tensor matrix = Tensor.from(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, dtype, 4, 3);
        Tensor view = matrix.slice(0, 1, 4, 2).transpose(0, 1);
        float[] row = view.rowData(1);
        assertArrayEquals(new float[]{5, 11}, row);
        row[0] = 99;
        assertEquals(5, matrix.get(1, 1));
        assertArrayEquals(new float[]{}, new Tensor(dtype, 2, 0).rowData(1));
        assertThrows(IndexOutOfBoundsException.class, () -> matrix.rowData(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> matrix.rowData(4));
        assertThrows(IllegalArgumentException.class, () -> new Tensor(dtype, 2, 3, 4).rowData(0));
        assertThrows(IllegalArgumentException.class, () -> new Tensor(dtype).rowData(0));
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void diagnosticPrintingUsesLogicalValuesAndDisplaysNonmatrixShapes(DType dtype) {
        Tensor matrix = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, dtype, 2, 3).transpose(0, 1);
        String printed = capture(() -> matrix.print("matrix"));
        assertTrue(printed.contains(String.format("%.4f %.4f ", 1.0f, 4.0f)));
        String nd = capture(() -> matrix.reshape(3, 1, 2).print("nd"));
        assertTrue(nd.contains(dtype + " shape=[3, 1, 2] values=[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]"));
        assertTrue(capture(() -> new Tensor(dtype).print("scalar")).contains("shape=[] values=[0.0]"));
        assertTrue(capture(() -> new Tensor(dtype, 0).print("empty")).contains("shape=[0] values=[]"));
    }

    @Test
    void denseFloatRowsPreserveNanPayloadsAndSignedZeros() {
        Tensor matrix = Tensor.from(new float[]{-0.0f, Float.intBitsToFloat(0x7fc12345)}, 1, 2);
        float[] values = matrix.rowData(0);
        assertEquals(0x80000000, Float.floatToRawIntBits(values[0]));
        assertEquals(0x7fc12345, Float.floatToRawIntBits(values[1]));
    }

    private static String capture(Runnable action) {
        var bytes = new ByteArrayOutputStream();
        PrintStream previous = System.out;
        try (var output = new PrintStream(bytes, true, StandardCharsets.UTF_8)) {
            System.setOut(output);
            action.run();
        } finally { System.setOut(previous); }
        return bytes.toString(StandardCharsets.UTF_8);
    }
}
