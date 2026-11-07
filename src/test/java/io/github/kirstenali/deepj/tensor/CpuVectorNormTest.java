package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class CpuVectorNormTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest @ValueSource(ints = {4096, 4097, 65539, 1247488})
    void boundedVectorReductionAgreesWithDoublePrecisionReference(int count) {
        assumeTrue(CpuBlas.AVAILABLE);
        float[] values = new float[count];
        var random = new Random(count);
        double squares = 0;
        for (int index = 0; index < count; index++) {
            values[index] = (random.nextFloat() - .5f) * .0001f;
            squares += (double) values[index] * values[index];
        }
        float expected = (float) Math.sqrt(squares);
        assertEquals(expected, Tensor.backend().l2Norm(List.of(Tensor.from(values, count))), expected * 1e-6f);
    }

    @Test void mixedRanksViewsAndRepeatedInputsRetainEveryLogicalValue() {
        Tensor dense = Tensor.ones(64, 65), view = dense.transpose(0, 1);
        Tensor offset = Tensor.ones(DType.FLOAT32, 4099).slice(0, 1, 4098);
        Tensor compact = Tensor.ones(DType.BFLOAT16, 129);
        List<Tensor> tensors = List.of(dense, view, dense, offset, compact, Tensor.ones(DType.FLOAT32, new int[0]));
        assertEquals((float) Math.sqrt(64 * 65 * 3 + 4097 + 129 + 1), Tensor.backend().l2Norm(tensors), 1e-5f);
        assertEquals(0, Tensor.backend().l2Norm(List.of()));
    }

    @Test void nonfiniteGradientsStillProduceANonfiniteNorm() {
        Tensor values = Tensor.ones(DType.FLOAT32, 4097);
        values.memory().set(4096, Float.POSITIVE_INFINITY);
        assertEquals(Float.POSITIVE_INFINITY, Tensor.backend().l2Norm(List.of(values)));
        values.memory().set(1024, Float.NaN);
        assertTrue(Float.isNaN(Tensor.backend().l2Norm(List.of(values))));
    }
}
