package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class ModelRmsTest {
    @BeforeEach void backend() {
        Tensor.setBackend(Boolean.getBoolean("gemma.test.metal") ? new MetalBackend() : new CpuBackend());
    }

    @ParameterizedTest @EnumSource(DType.class)
    void inferenceRetainsFp32StatisticsAndModelSpecificAffinePrecision(DType dtype) {
        Tensor input = values(dtype, 4, 2304).slice(0, 1, 4);
        Tensor gain = values(dtype, 2304), bias = values(DType.FLOAT32, 2304);
        try (var ignored = GradMode.noGrad()) {
            for (boolean gemma : new boolean[]{false, true}) {
                checkInference(input, gain, bias, gemma);
                checkInference(input, gain, null, gemma);
            }
        }
    }

    private static void checkInference(Tensor input, Tensor gain, Tensor bias, boolean gemma) {
        Tensor expected = composed(input, gain, bias, gemma);
        assertArrayEquals(expected.toFloatArray(), ModelOps.rms(input, gain, bias, 1e-6f, gemma).toFloatArray());
        Tensor actual = ModelInference.batch(() -> ModelOps.rms(input, gain, bias, 1e-6f, gemma));
        assertCompactRounding(expected.toFloatArray(), actual.toFloatArray(), input, gain, gemma);
        assertEquals(input.dtype(), actual.dtype());
        assertFalse(actual.requiresGrad());
    }

    @ParameterizedTest @EnumSource(DType.class)
    void trainingKeepsTheExistingForwardAndEveryGradient(DType dtype) {
        Tensor input = values(dtype, 2, 16).requiresGrad(true), gain = values(dtype, 16).requiresGrad(true);
        Tensor bias = values(DType.FLOAT32, 16).requiresGrad(true);
        Tensor expected = composed(input, gain, bias, true), actual = ModelOps.rms(input, gain, bias, 1e-6f, true);
        assertArrayEquals(expected.toFloatArray(), actual.toFloatArray());
        float[][] gradients = gradients(expected, input, gain, bias);
        java.util.stream.Stream.of(input, gain, bias).forEach(Tensor::zeroGrad);
        float[][] actualGradients = gradients(actual, input, gain, bias);
        for (int index = 0; index < gradients.length; index++) {
            assertArrayEquals(gradients[index], actualGradients[index]);
        }
    }

    private static float[][] gradients(Tensor output, Tensor... inputs) {
        output.sumTensor().backward();
        return java.util.Arrays.stream(inputs).map(input -> input.grad().toFloatArray()).toArray(float[][]::new);
    }

    private static void assertCompactRounding(float[] expected, float[] actual, Tensor input, Tensor gain,
                                               boolean gemma) {
        DType dtype = input.dtype();
        float[] values = normalized(input).toFloatArray(), weights = gain.toFloatArray();
        for (int index = 0; index < expected.length; index++) {
            float tolerance = compactUlp(expected[index], dtype);
            if (!gemma && dtype != DType.FLOAT32) {
                // Early compact rounding is scaled before bias cancellation, so bound that intermediate too.
                float weight = weights[index % weights.length];
                tolerance += compactUlp(values[index], dtype) * Math.abs(weight)
                        + compactUlp(values[index] * weight, dtype);
            }
            assertEquals(expected[index], actual[index], dtype == DType.FLOAT32 ? 0 : Math.max(2e-6f, tolerance));
        }
    }

    private static float compactUlp(float value, DType dtype) {
        int mantissa = dtype == DType.BFLOAT16 ? 7 : 10, minimum = dtype == DType.BFLOAT16 ? -126 : -14;
        return Math.scalb(1f, Math.max(minimum, Math.getExponent(value)) - mantissa);
    }

    private static Tensor normalized(Tensor input) {
        Tensor value = input.to(DType.FLOAT32);
        return value.divide(value.multiply(value).mean(-1, true).addScalar(1e-6f).sqrt());
    }

    private static Tensor composed(Tensor input, Tensor gain, Tensor bias, boolean gemma) {
        Tensor result = normalized(input);
        if (!gemma && gain.dtype() != DType.FLOAT32) result = result.to(gain.dtype());
        result = result.multiply(gemma ? gain.to(DType.FLOAT32).addScalar(1) : gain);
        return (bias == null ? result : result.add(bias)).to(input.dtype());
    }

    private static Tensor values(DType dtype, int... shape) {
        int count = java.util.Arrays.stream(shape).reduce(1, Math::multiplyExact);
        float[] data = new float[count];
        var random = new Random(12);
        for (int index = 0; index < count; index++) { data[index] = random.nextFloat() - .5f; }
        return Tensor.from(data, dtype, shape);
    }
}
