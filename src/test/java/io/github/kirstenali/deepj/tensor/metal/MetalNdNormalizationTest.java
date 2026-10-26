package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.autograd.NormalizationOps;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class MetalNdNormalizationTest extends MetalDifferentialTestSupport {
    private static final float EPSILON = 1e-5f;

    @Test
    void allRanksAndDtypesMatchIndependentDoubleStatistics() {
        for (int rank = 1; rank <= 6; rank++) {
            for (DType dtype : DType.values()) {
                int[] shape = new int[rank];
                Arrays.fill(shape, 1);
                shape[rank - 1] = 3;
                checkReference(Tensor.from(new float[]{-0.3f, 0.7f, 1.2f}, dtype, shape), dtype, true);
                checkReference(Tensor.from(new float[]{-0.3f, 0.7f, 1.2f}, dtype, shape), dtype, false);
            }
        }
    }

    private void checkReference(Tensor input, DType dtype, boolean centered) {
        Tensor gamma = Tensor.from(new float[]{0.7f, 1.1f, -0.4f}, dtype);
        Tensor beta = Tensor.from(new float[]{0.1f, -0.2f, 0.3f}, dtype);
        float[] expected = reference(input.toFloatArray(), gamma.toFloatArray(), beta.toFloatArray(), centered);
        var actual = metal.normalize(input, gamma, centered ? beta : null, EPSILON, centered);
        assertArrayEquals(Tensor.from(expected, dtype).toFloatArray(), actual.output().toFloatArray(), 3e-6f);
        assertEquals(DType.FLOAT32, actual.normalized().dtype());
        assertEquals(DType.FLOAT32, actual.denominator().dtype());
        assertArrayEquals(input.shape(), actual.output().shape());
    }

    private static float[] reference(float[] input, float[] gain, float[] bias, boolean centered) {
        double mean = 0, squares = 0;
        for (float value : input) { mean += value; }
        mean = centered ? mean / input.length : 0;
        for (float value : input) { squares += (value - mean) * (value - mean); }
        double denominator = Math.sqrt(squares / input.length + EPSILON);
        float[] output = new float[input.length];
        for (int index = 0; index < output.length; index++) {
            output[index] = (float) ((input[index] - mean) / denominator * gain[index] + (centered ? bias[index] : 0));
        }
        return output;
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void stridedMixedParametersMatchCpuAndLargeCompactValuesUseFloat32Statistics(DType dtype) {
        Tensor base = Tensor.from(new float[]{9, 8, 7, 1, 2, 3, 4, 5, 6, 6, 5, 4}, dtype, 2, 2, 3);
        Tensor input = base.transpose(0, 1).slice(0, 1, 2, 1);
        Tensor gain = Tensor.from(new float[]{2, 9, 3, 9, 4, 9}, dtype).slice(0, 0, 6, 2);
        Tensor bias = Tensor.from(new float[]{0.1f, 0.2f, 0.3f});
        assertArrayEquals(cpu.normalize(input, gain, bias, EPSILON, true).output().toFloatArray(),
                metal.normalize(input, gain, bias, EPSILON, true).output().toFloatArray(), 3e-6f);
        var rms = metal.rmsNorm(Tensor.from(new float[]{200, 300}, dtype), Tensor.from(new float[]{1, 1}, dtype),
                EPSILON);
        assertTrue(Float.isFinite(rms.output().get(0)));
        assertEquals(DType.FLOAT32, rms.rms().dtype());
        assertEquals((float) Math.sqrt(65000 + EPSILON), rms.rms().item(), 3e-5f);
    }

    @Test
    void contiguousFloat32NdRmsUsesTheSameFusedMatrixKernel() {
        Tensor input = Tensor.from(new float[]{0.1f, 0.3f, -0.5f, 0.2f, 0.7f, 0.4f}, 2, 1, 3);
        Tensor gain = Tensor.from(new float[]{0.9f, -0.4f, 0.5f});
        var matrix = metal.rmsNorm(input.reshape(2, 3), gain.reshape(1, 3), EPSILON);
        var actual = metal.normalize(input, gain, null, EPSILON, false);
        assertArrayEquals(matrix.output().toFloatArray(), actual.output().toFloatArray());
        assertArrayEquals(matrix.normalized().toFloatArray(), actual.normalized().toFloatArray());
        assertArrayEquals(matrix.rms().toFloatArray(), actual.denominator().toFloatArray());
        Tensor upstream = Tensor.from(new float[]{0.2f, 0.4f, -0.1f, 0.3f, -0.2f, 0.5f}, input.shape());
        Tensor expected = metal.rmsNormBackward(upstream.reshape(2, 3), matrix.normalized(), matrix.rms(),
                gain.reshape(1, 3));
        assertArrayEquals(expected.toFloatArray(), metal.normalizationBackward(upstream, actual.normalized(),
                actual.denominator(), gain, false).toFloatArray());
    }

    @Test
    void contiguousFloat32NdLayerNormKeepsTheOriginalMatrixOperationBits() {
        Tensor input = Tensor.from(new float[]{0.2f, -0.4f, 0.5f, -0.1f, 0.3f, 0.7f}, 2, 1, 3);
        Tensor gamma = Tensor.from(new float[]{0.5f, -0.1f, 0.8f}), beta = Tensor.from(new float[]{0.2f, 0.3f, -0.2f});
        Tensor expected = NormalizationOps.layerNorm(input.reshape(2, 3), gamma.reshape(1, 3),
                beta.reshape(1, 3), EPSILON);
        var result = metal.normalize(input, gamma, beta, EPSILON, true);
        assertArrayEquals(expected.toFloatArray(), result.output().toFloatArray());
        Tensor upstream = Tensor.from(new float[]{0.1f, 0.4f, -0.2f, 0.3f, -0.1f, 0.5f}, input.shape());
        Tensor scaled = metal.multiplyBroadcastRows(upstream.reshape(2, 3), gamma.reshape(1, 3));
        Tensor gradient = metal.layerNormBackward(scaled, result.normalized().reshape(2, 3),
                result.denominator().reshape(2, 1), 3);
        assertArrayEquals(gradient.toFloatArray(), metal.normalizationBackward(upstream,
                result.normalized(), result.denominator(), gamma, true).toFloatArray());
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void gpuSavedValuesStayHostLazyAndLeadingEmptyAxesAllocateNothing(DType dtype) {
        Tensor input = metal.copy(Tensor.from(new float[]{0.1f, 0.5f, 0.8f}, dtype, 1, 1, 3));
        Tensor gain = metal.copy(Tensor.from(new float[]{1, 2, 3}, dtype));
        var result = metal.normalize(input, gain, null, EPSILON, false);
        metal.graph.flush();
        assertFalse(TensorTestStorage.hasHostStorage(input));
        assertFalse(TensorTestStorage.hasHostStorage(result.output()));
        assertFalse(TensorTestStorage.hasHostStorage(result.normalized()));
        assertFalse(TensorTestStorage.hasHostStorage(result.denominator()));
        var empty = metal.normalize(new Tensor(dtype, 0, 2, 3), gain, null, EPSILON, false);
        assertArrayEquals(new int[]{0, 2, 1}, empty.denominator().shape());
        assertEquals(0, empty.output().storageBytes());
    }
}
