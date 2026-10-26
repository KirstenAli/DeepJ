package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

class CpuRowAccessTest {
    private final CpuBackend backend = new CpuBackend();
    @BeforeEach void cpu() { Tensor.setBackend(backend); }

    @ParameterizedTest
    @EnumSource(DType.class)
    void convenienceAccessorsUseActualTypedStridedCoordinatesAndIndependentCopies(DType dtype) {
        Tensor parent = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, dtype, 2, 3);
        Tensor matrix = parent.permute(1, 0);
        assertEquals(5, backend.get(matrix, 1, 1));
        backend.set(matrix, 0, 1, 0.25f);
        assertEquals(0.25f, parent.get(1, 0));
        Tensor row = backend.getRow(matrix, 1);
        assertEquals(dtype, row.dtype());
        assertArrayEquals(new float[]{2, 5}, row.toFloatArray());
        row.set(0, 0, 9);
        assertEquals(2, matrix.get(1, 0));
        assertArrayEquals(new float[]{3, 1}, backend.sliceRows(matrix, new int[]{2, 0}, 1).toFloatArray());
        assertArrayEquals(new float[]{3, 6, 1, 0.25f}, backend.sampleRows(matrix, 2, new Random(42)).toFloatArray());
        backend.setRow(matrix, 0, matrix, 2);
        assertArrayEquals(new float[]{3, 6}, backend.getRow(matrix, 0).toFloatArray());
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void facadeRowAssignmentPreservesParentMutationFenceAndTrackedAliasVersion(DType dtype) {
        Tensor parent = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, dtype, 2, 3).requiresGrad(true);
        Tensor matrix = parent.permute(1, 0);
        Tensor loss = parent.multiply(parent).sumTensor();
        long version = parent.gradientState().version();
        assertThrows(IllegalStateException.class, () -> matrix.setRow(0, new Tensor(dtype, 1, 2), 0));
        try (var ignored = GradMode.noGrad()) { matrix.setRow(0, new Tensor(dtype, 1, 2), 0); }
        assertTrue(parent.gradientState().version() > version);
        assertThrows(IllegalStateException.class, loss::backward);
    }

    @Test
    void invalidRanksRowsColumnsCountsAndBroadcastWritesHaveExplicitChecks() {
        Tensor matrix = new Tensor(DType.FLOAT16, 2, 3);
        assertThrows(IndexOutOfBoundsException.class, () -> backend.getRow(matrix, 2));
        assertThrows(IndexOutOfBoundsException.class, () -> backend.setRow(matrix, -1, matrix, 0));
        assertThrows(IllegalArgumentException.class, () -> backend.setRow(matrix, 0, new Tensor(1, 2), 0));
        assertThrows(IllegalArgumentException.class, () -> backend.sliceRows(matrix, new int[]{0}, 4));
        assertThrows(IllegalArgumentException.class, () -> backend.sliceRows(matrix, new int[]{0}, -1));
        assertThrows(IllegalArgumentException.class, () -> backend.getRow(new Tensor(2, 2, 3), 0));
        assertThrows(IllegalArgumentException.class, () -> backend.sampleRows(matrix, -1, new Random(1)));
        assertArrayEquals(new int[]{0, 3}, backend.sampleRows(matrix, 0, new Random(1)).shape());
        Tensor overlap = new Tensor(1, 3).expand(2, 3);
        assertThrows(IllegalStateException.class, () -> backend.setRow(overlap, 0, matrix, 0));
    }
}
