package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.autograd.LayoutOps;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.testing.PositionReference;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class CpuTypedPositionTest {
    private final CpuBackend backend = new CpuBackend();
    @BeforeEach void cpu() { Tensor.setBackend(backend); }

    @ParameterizedTest
    @EnumSource(DType.class)
    void stridedCausalAndMixedTableRotaryMatchIndependentQuantizedEquations(DType dtype) {
        Tensor input = PositionReference.input(dtype), cosine = PositionReference.table(DType.FLOAT32, false);
        Tensor sine = PositionReference.table(DType.FLOAT32, true);
        assertArrayEquals(PositionReference.mask(input, 4), backend.causalMask(input, 4).toFloatArray());
        assertArrayEquals(PositionReference.softmax(input, 4, 0.625f), backend.causalSoftmax(input, 4, 0.625f)
                .toFloatArray(), 3e-6f);
        for (boolean inverse : new boolean[]{false, true}) {
            Tensor output = backend.rotary(input, cosine, sine, 4, inverse);
            assertEquals(dtype, output.dtype());
            assertArrayEquals(PositionReference.rotary(input, cosine, sine, 4, inverse), output.toFloatArray(), 3e-6f);
        }
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void maskedFutureNonfiniteValuesAreNotReadAndEmptyAxesStayEmpty(DType dtype) {
        Tensor source = Tensor.from(new float[]{0, Float.NaN, 0, 1}, dtype, 2, 2).permute(1, 0).permute(1, 0);
        Tensor output = CpuNdPosition.softmax(source, 2, 1);
        assertArrayEquals(new float[]{1, 0}, new float[]{output.get(0, 0), output.get(0, 1)});
        assertEquals(0, backend.causalMask(new Tensor(dtype, 0, 4), 4).numel());
        assertEquals(0, backend.causalSoftmax(new Tensor(dtype, 0, 4), 4, 1).numel());
        assertEquals(0, backend.rotary(new Tensor(dtype, 4, 0), new Tensor(4, 0), new Tensor(4, 0), 4, false).numel());
    }

    @Test
    void invalidPositionShapesAndConstantsAreRejected() {
        Tensor input = new Tensor(DType.FLOAT16, 8, 4), table = new Tensor(4, 2);
        assertThrows(IllegalArgumentException.class, () -> backend.causalSoftmax(input, 3, 1));
        assertThrows(IllegalArgumentException.class, () -> backend.causalSoftmax(input, 4, Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> backend.rotary(input, table, table, 3, false));
        assertThrows(IllegalArgumentException.class, () -> backend.rotary(input, new Tensor(3, 2), table, 4, false));
        assertThrows(IllegalArgumentException.class,
                () -> LayoutOps.rotary(input, table.requiresGrad(true), table, 4, false));
    }
}
