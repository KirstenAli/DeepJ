package io.github.kirstenali.deepj.tensor.cpu;

import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.Random;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class CpuFusedRmsBackwardTest {
    @ParameterizedTest @CsvSource({"1,7", "1,128", "64,256", "256,128", "128,2048"})
    void fusedBackwardMatchesOriginalCompositionExactly(int rows, int width) {
        CpuBackend backend = new CpuBackend();
        Tensor.setBackend(backend);
        Tensor input = values(rows, width), gradient = values(rows, width), gamma = values(1, width);
        var normalized = backend.rmsNorm(input, gamma, 1e-6f);
        Tensor scaled = backend.multiplyBroadcastRows(gradient, gamma);
        Tensor inner = backend.meanAlongRows(backend.multiply(scaled, normalized.normalized()));
        Tensor expected = backend.divideBroadcastCols(backend.subtract(scaled,
                backend.multiplyBroadcastCols(normalized.normalized(), inner)), normalized.rms());
        Tensor actual = backend.rmsNormBackward(gradient, normalized.normalized(), normalized.rms(), gamma);
        assertArrayEquals(expected.toFloatArray(), actual.toFloatArray());
    }

    private static Tensor values(int rows, int width) {
        var random = new Random(137L * rows + width);
        float[] values = new float[rows * width];
        for (int index = 0; index < values.length; index++) { values[index] = random.nextFloat() * 8 - 4; }
        return Tensor.from(values, rows, width);
    }
}
