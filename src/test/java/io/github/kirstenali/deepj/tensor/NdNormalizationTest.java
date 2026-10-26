package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.autograd.NormalizationOps;
import io.github.kirstenali.deepj.layers.transformer.norm.LayerNorm1D;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class NdNormalizationTest {
    private static final float EPSILON = 1e-5f;
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void allRanksAndDtypesMatchIndependentFinalAxisNormalization() {
        for (DType type : DType.values()) {
            for (int rank = 1; rank <= 6; rank++) {
                int[] shape = new int[rank];
                java.util.Arrays.fill(shape, 1);
                shape[rank - 1] = 3;
                Tensor input = Tensor.from(new float[]{-0.3f, 0.7f, 1.2f}, type, shape);
                checkReference(input, type, true);
                checkReference(input, type, false);
            }
        }
    }

    private static void checkReference(Tensor input, DType type, boolean centered) {
        Tensor gamma = Tensor.from(new float[]{0.7f, 1.1f, -0.4f}, type);
        Tensor beta = Tensor.from(new float[]{0.1f, -0.2f, 0.3f}, type);
        var result = Tensor.backend().normalize(input, gamma, centered ? beta : null, EPSILON, centered);
        float[] expected = reference(input.toFloatArray(), gamma.toFloatArray(), beta.toFloatArray(), centered);
        assertArrayEquals(Tensor.from(expected, type).toFloatArray(), result.output().toFloatArray(), 2e-7f);
        assertArrayEquals(input.shape(), result.output().shape());
        assertEquals(DType.FLOAT32, result.normalized().dtype());
        assertEquals(DType.FLOAT32, result.denominator().dtype());
    }

    private static float[] reference(float[] input, float[] gamma, float[] beta, boolean centered) {
        double mean = 0, squares = 0;
        for (float value : input) {
            mean += value;
        }
        mean = centered ? mean / input.length : 0;
        for (float value : input) {
            squares += (value - mean) * (value - mean);
        }
        double denominator = Math.sqrt(squares / input.length + EPSILON);
        float[] result = new float[input.length];
        for (int index = 0; index < result.length; index++) {
            result[index] = (float) ((input[index] - mean) / denominator * gamma[index]
                    + (centered ? beta[index] : 0));
        }
        return result;
    }

    @Test
    void stridedAndOffsetLayoutsKeepSeparateLeadingPositions() {
        Tensor base = Tensor.from(new float[]{9, 8, 7, 1, 2, 3, 4, 5, 6, 6, 5, 4}, 2, 2, 3);
        Tensor input = base.transpose(0, 1).slice(0, 1, 2, 1);
        Tensor gain = Tensor.from(new float[]{2, 3, 4}, 1, 3), bias = Tensor.from(new float[]{1, 1, 1}, 1, 3);
        Tensor matrix = Tensor.from(input.toFloatArray(), 2, 3);
        assertArrayEquals(NormalizationOps.layerNorm(matrix, gain, bias, EPSILON).toFloatArray(),
                NormalizationOps.layerNorm(input, gain, bias, EPSILON).toFloatArray());
        assertArrayEquals(NormalizationOps.rmsNorm(matrix, gain, EPSILON).toFloatArray(),
                NormalizationOps.rmsNorm(input, gain, EPSILON).toFloatArray());
    }

    @Test
    void compactRmsStatisticsDoNotOverflowInCompactIntermediates() {
        Tensor input = Tensor.from(new float[]{200, 300}, DType.FLOAT16);
        Tensor gain = Tensor.from(new float[]{1, 1}, DType.FLOAT16);
        RmsNormResult result = Tensor.backend().rmsNorm(input, gain, EPSILON);
        assertEquals((float) Math.sqrt(65000 + EPSILON), result.rms().item());
        assertTrue(Float.isFinite(result.output().get(0)));
        assertTrue(Float.isFinite(result.output().get(1)));
        assertEquals(4, result.output().storageBytes());
        assertEquals(DType.FLOAT16, result.output().dtype());
    }

    @Test
    void layersAcceptVectorsAndNdBatchesWithoutChangingWidth() {
        var layer = new LayerNorm1D(3);
        var rms = new RMSNorm1D(3);
        Tensor vector = Tensor.from(new float[]{1, 2, 3});
        assertArrayEquals(new int[]{3}, layer.forward(vector).shape());
        assertArrayEquals(new int[]{2, 1, 3}, rms.forward(new Tensor(2, 1, 3)).shape());
        assertThrows(IllegalArgumentException.class, () -> layer.forward(new Tensor(2, 3, 4)));
        assertThrows(IllegalArgumentException.class, () -> rms.forward(new Tensor(new int[0])));
    }

    @Test
    void invalidNormalizationMetadataIsRejectedExplicitly() {
        Tensor input = new Tensor(2, 3), gamma = new Tensor(3), beta = new Tensor(3);
        assertThrows(IllegalArgumentException.class, () -> Tensor.backend().normalize(
                new Tensor(0), gamma, beta, EPSILON, true));
        assertThrows(IllegalArgumentException.class, () -> Tensor.backend().normalize(
                input, new Tensor(2, 3), beta, EPSILON, true));
        assertThrows(IllegalArgumentException.class, () -> Tensor.backend().normalize(
                input, new Tensor(2), beta, EPSILON, true));
        assertThrows(IllegalArgumentException.class, () -> Tensor.backend().normalize(
                input, gamma, beta, Float.NaN, true));
        assertThrows(IllegalArgumentException.class, () -> Tensor.backend().normalize(
                input, gamma, beta, 0, true));
        assertThrows(IllegalArgumentException.class, () -> Tensor.backend().normalize(
                input, gamma, beta, EPSILON, false));
    }

    @Test
    void matrixAndGeneralEntryPointsShareTheSameEpsilonAndAffinePreflight() {
        Tensor input = new Tensor(2, 3), gamma = new Tensor(1, 3), beta = new Tensor(1, 3);
        assertThrows(IllegalArgumentException.class, () -> Tensor.backend().rmsNorm(input, gamma, 0));
        assertThrows(IllegalArgumentException.class, () -> NormalizationOps.rmsNorm(input, gamma, Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> NormalizationOps.layerNorm(input, gamma, beta, -1));
        assertThrows(IllegalArgumentException.class,
                () -> NormalizationOps.layerNorm(input, gamma, new Tensor(2, 3), EPSILON));
    }
}
