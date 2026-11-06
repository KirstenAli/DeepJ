package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class CpuNormalizationRowsTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test void matrixRmsFusionPreservesEveryOutputStatisticAndBackwardBit() {
        for (int rows : new int[]{1, 64}) {
            Tensor source = input(DType.FLOAT32, rows, 257), gain = input(DType.FLOAT32, 1, 257);
            Tensor gradient = input(DType.FLOAT32, rows, 257);
            var backend = Tensor.backend();
            Tensor squares = backend.multiply(source, source);
            Tensor rms = backend.sqrt(backend.addScalar(backend.meanAlongRows(squares), 1e-5f));
            Tensor normalized = backend.divideBroadcastCols(source, rms);
            Tensor expected = backend.multiplyBroadcastRows(normalized, gain);
            var actual = backend.rmsNorm(source, gain, 1e-5f);
            assertArrayEquals(expected.toFloatArray(), actual.output().toFloatArray());
            assertArrayEquals(normalized.toFloatArray(), actual.normalized().toFloatArray());
            assertArrayEquals(rms.toFloatArray(), actual.rms().toFloatArray());
            assertArrayEquals(backend.rmsNormBackward(gradient, normalized, rms, gain).toFloatArray(),
                    backend.rmsNormBackward(gradient, actual.normalized(), actual.rms(), gain).toFloatArray());
        }
    }

    @ParameterizedTest @EnumSource(DType.class)
    void parallelRowsPreserveStatisticsAndEveryOutputBitForStridedAndDenseInputs(DType dtype) {
        Tensor source = input(dtype, 4, 16, 257);
        for (boolean centered : new boolean[]{false, true}) {
            check(source, dtype, centered);
            check(source.permute(1, 0, 2), dtype, centered);
        }
    }

    private static void check(Tensor input, DType dtype, boolean centered) {
        boolean previous = DeepJExecutor.isParallelEnabled();
        Tensor gain = input(dtype, 257), bias = centered ? input(dtype, 257) : null;
        try {
            DeepJExecutor.setParallelEnabled(false);
            var expected = Tensor.backend().normalize(input, gain, bias, 1e-5f, centered);
            DeepJExecutor.setParallelEnabled(true);
            var actual = Tensor.backend().normalize(input, gain, bias, 1e-5f, centered);
            assertArrayEquals(expected.output().toFloatArray(), actual.output().toFloatArray());
            assertArrayEquals(expected.normalized().toFloatArray(), actual.normalized().toFloatArray());
            assertArrayEquals(expected.denominator().toFloatArray(), actual.denominator().toFloatArray());
        } finally {
            DeepJExecutor.setParallelEnabled(previous);
        }
    }

    private static Tensor input(DType dtype, int... shape) {
        var random = new Random(20261106);
        float[] values = new float[new TensorShape(shape).elements()];
        for (int i = 0; i < values.length; i++) { values[i] = random.nextFloat() * 8 - 4; }
        return Tensor.from(values, dtype, shape);
    }
}
