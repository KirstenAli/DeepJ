package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class CpuCompactSoftmaxTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void denseLastAxisSoftmaxMatchesStridedScalarFallback(DType type) {
        for (int width : new int[]{1, 7, 64, 128, 129, 257, 4096, 4097}) {
            Tensor strided = values(type, 4, width);
            Tensor dense = Tensor.from(strided.toFloatArray(), type, strided.shape());
            assertArrayEquals(CpuNdSoftmax.apply(strided, 1).toFloatArray(),
                    CpuNdSoftmax.apply(dense, 1).toFloatArray());
        }
    }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void causalRowsRetainMaskedZerosAndSpecialValues(DType type) {
        for (int width : new int[]{64, 129, 257}) {
            Tensor strided = values(type, width * 2, width);
            Tensor dense = Tensor.from(strided.toFloatArray(), type, strided.shape());
            assertArrayEquals(CpuNdPosition.softmax(strided, width, .125f).toFloatArray(),
                    CpuNdPosition.softmax(dense, width, .125f).toFloatArray());
        }
    }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void finiteRowsAndRoundingFallbacksPreserveProbabilities(DType type) {
        Tensor strided = values(type, 2048, 64).slice(0, 4, 2048);
        Tensor dense = Tensor.from(strided.toFloatArray(), type, strided.shape());
        assertArrayEquals(CpuNdSoftmax.apply(strided, 1).toFloatArray(),
                CpuNdSoftmax.apply(dense, 1).toFloatArray());
    }

    private static Tensor values(DType type, int rows, int width) {
        float[] data = new float[rows * width * 2];
        var random = new Random(181 + width);
        for (int index = 0; index < data.length; index++) { data[index] = (float) random.nextGaussian() * 4; }
        data[0] = Float.NaN;
        data[width * 2] = Float.NEGATIVE_INFINITY;
        data[width * 4] = Float.POSITIVE_INFINITY;
        return Tensor.from(data, type, rows, width * 2).slice(1, 0, width * 2, 2);
    }
}
