package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorTestStorage;
import java.util.Random;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class MetalComposedRmsTest extends MetalDifferentialTestSupport {
    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void compactAffineRoundsBeforeBiasCancellationInDenseAndChannelLayouts(DType dtype) {
        try (var ignored = GradMode.noGrad()) {
            Tensor gain = Tensor.from(new float[]{1.5f, 2.5f}, dtype, 2);
            cancelled(Tensor.from(new float[]{1, 3}, dtype, 2), gain);
            cancelled(Tensor.from(new float[]{1, 1, 3, 3}, dtype, 1, 2, 1, 2).permute(0, 2, 3, 1), gain);
        }
    }

    private void cancelled(Tensor input, Tensor gain) {
        Tensor value = input.to(DType.FLOAT32);
        Tensor denominator = value.multiply(value).mean(-1, true).addScalar(1e-5f).sqrt();
        Tensor product = value.divide(denominator).to(input.dtype()).multiply(gain);
        Tensor bias = product.reshape(-1, 2).select(0, 0).neg();
        Tensor output = metal.storedRmsForInference(input, gain, bias, 1e-5f);
        assertArrayEquals(new float[input.numel()], output.toFloatArray(), 0);
    }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void parallelStoredRmsPreservesCompactAffinePrecisionForWideRows(DType dtype) {
        try (var ignored = GradMode.noGrad()) {
            Tensor input = values(dtype, 2, 3, 256), gain = values(dtype, 256);
            Tensor value = input.to(DType.FLOAT32);
            Tensor denominator = value.multiply(value).mean(-1, true).addScalar(1e-5f).sqrt();
            Tensor expected = value.divide(denominator).to(dtype).multiply(gain);
            assertArrayEquals(expected.toFloatArray(), metal.storedRmsForInference(input, gain, null, 1e-5f)
                    .toFloatArray(), dtype == DType.FLOAT16 ? 0.002f : 0.02f);
        }
    }

    @ParameterizedTest @EnumSource(DType.class)
    void channelNormalizationPreservesUnfusedFloat32RoundingIncludingOffsetsAndStridedParameters(DType dtype) {
        try (var ignored = GradMode.noGrad()) {
            for (int width : new int[]{3, 32, 128, 256}) {
                Tensor input = values(dtype, 3, width, 2, 3).slice(0, 1, 3).permute(0, 2, 3, 1);
                Tensor gain = values(DType.FLOAT32, width * 2 + 1).slice(0, 1, width * 2 + 1, 2);
                Tensor bias = values(dtype, width * 2 + 1).slice(0, 1, width * 2 + 1, 2);
                compare(input, gain, null);
                compare(input, gain, bias);
            }
            compare(values(dtype, 2, 3, 32), Tensor.ones(DType.FLOAT32, 32), null);
            compare(values(dtype, 2, 3, 4, 8), Tensor.ones(DType.FLOAT32, 8), null);
        }
    }

    private void compare(Tensor input, Tensor gain, Tensor bias) {
        Tensor value = input.to(DType.FLOAT32);
        Tensor denominator = value.multiply(value).mean(-1, true).addScalar(1e-5f).sqrt();
        Tensor expected = value.divide(denominator).multiply(gain);
        if (bias != null) expected = expected.add(bias.to(DType.FLOAT32));
        float[] reference = expected.to(input.dtype()).toFloatArray();
        Tensor result = metal.composedRmsForInference(input, gain, bias, 1e-5f);
        assertFalse(TensorTestStorage.hasHostStorage(result));
        assertEquals(input.dtype(), result.dtype());
        if (input.rank() == 4 && input.strides()[3] != 1) assertArrayEquals(input.strides(), result.strides());
        assertArrayEquals(reference, result.toFloatArray(), 0);
    }

    private static Tensor values(DType dtype, int... shape) {
        int count = java.util.Arrays.stream(shape).reduce(1, Math::multiplyExact);
        float[] values = new float[count];
        Random random = new Random(count + 239);
        for (int index = 0; index < count; index++) { values[index] = 4 * random.nextFloat() - 2; }
        return Tensor.from(values, dtype, shape);
    }
}
