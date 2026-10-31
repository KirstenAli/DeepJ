package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class NdNormTest {
    private final CpuBackend cpu = new CpuBackend();

    @ParameterizedTest
    @EnumSource(DType.class)
    void normsDecodeLogicalStridesAndCompactValuesAndAccumulateAcrossTensors(DType dtype) {
        Tensor source = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, dtype, 2, 3);
        Tensor view = TensorViews.slice(TensorViews.permute(source, 1, 0), 0, 0, 3, 2);
        Tensor scalar = Tensor.from(new float[]{2}, dtype, new int[0]);
        assertEquals((float) Math.sqrt(1 + 16 + 9 + 36 + 4), cpu.l2Norm(List.of(view, scalar)));
        assertEquals((float) Math.sqrt(91), cpu.l2Norm(List.of(source)));
        assertEquals(0, cpu.l2Norm(List.of(new Tensor(dtype, 2, 0, 3))));
    }

    @Test
    void fullFloat32OrderIsTheSameAsTheOriginalLoop() {
        Tensor first = Tensor.from(new float[]{2048, 1, 1}, 1, 3), second = Tensor.from(new float[]{-2, 3}, 1, 2);
        float squares = 0;
        for (Tensor input : List.of(first, second)) {
            for (float value : input.rawData()) { squares += value * value; }
        }
        assertEquals((float) Math.sqrt(squares), cpu.l2Norm(List.of(first, second)));
        assertEquals(0, cpu.l2Norm(List.of()));
    }
}
