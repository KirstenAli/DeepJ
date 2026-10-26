package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.NormalizationOps;
import io.github.kirstenali.deepj.tensor.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertTensorGradients;
import static org.junit.jupiter.api.Assertions.*;

class MetalNdNormalizationGradientTest extends MetalDifferentialTestSupport {
    private static final float EPSILON = 1e-5f;

    @Test
    void centeredStridedInputAndVectorAffineParametersMatchActualGpuFiniteDifferences() {
        Tensor input = input(), gain = gain(), bias = Tensor.from(new float[]{0.2f, -0.1f, 0.4f});
        Tensor upstream = upstream();
        assertTensorGradients(input, value -> NormalizationOps.layerNorm(value.transpose(0, 1), gain, bias, EPSILON),
                upstream.transpose(0, 1), 1e-3f, 2e-4f, 2e-3f);
        assertTensorGradients(gain, value -> NormalizationOps.layerNorm(input, value, bias, EPSILON),
                upstream, 1e-3f, 2e-4f, 2e-3f);
        assertTensorGradients(bias, value -> NormalizationOps.layerNorm(input, gain, value, EPSILON),
                upstream, 1e-3f, 2e-4f, 2e-3f);
    }

    @Test
    void rmsInputAndGainHaveFiniteDifferenceAdjoints() {
        Tensor input = input(), gain = gain();
        assertTensorGradients(input, value -> NormalizationOps.rmsNorm(value, gain, EPSILON),
                upstream(), 1e-3f, 2e-4f, 2e-3f);
        assertTensorGradients(gain, value -> NormalizationOps.rmsNorm(input, value, EPSILON),
                upstream(), 1e-3f, 2e-4f, 2e-3f);
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void savedFloat32StatisticsAndInputAdjointsMatchDecodedCpuArithmetic(DType dtype) {
        Tensor input = input().to(dtype), gamma = gain().to(dtype), bias = new Tensor(dtype, 3);
        for (boolean centered : new boolean[]{false, true}) {
            var expected = cpu.normalize(input, gamma, centered ? bias : null, EPSILON, centered);
            var actual = metal.normalize(input, gamma, centered ? bias : null, EPSILON, centered);
            Tensor reference = cpu.normalizationBackward(upstream(), expected.normalized(),
                    expected.denominator(), gamma, centered);
            Tensor gradient = metal.normalizationBackward(upstream(), actual.normalized(),
                    actual.denominator(), gamma, centered);
            assertArrayEquals(reference.toFloatArray(), gradient.toFloatArray(), 3e-6f);
            assertEquals(DType.FLOAT32, gradient.dtype());
        }
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void compactAffineLeavesGetFloat32ParameterAndInputGradients(DType dtype) {
        Tensor input = input().to(dtype).requiresGrad(true), gain = gain().to(dtype).requiresGrad(true);
        Tensor bias = Tensor.from(new float[]{0.2f, 0.3f, -0.1f}, dtype, 1, 3).requiresGrad(true);
        NormalizationOps.layerNorm(input, gain, bias, EPSILON).backward(upstream());
        assertEquals(DType.FLOAT32, input.grad().dtype());
        assertEquals(DType.FLOAT32, gain.grad().dtype());
        assertArrayEquals(new int[]{1, 3}, bias.grad().shape());
        assertArrayEquals(new float[]{0.6f, 0.1f, 0.1f}, bias.grad().toFloatArray(), 1e-7f);
        Tensor empty = new Tensor(dtype, 0, 2, 3).requiresGrad(true);
        empty.zeroGrad();
        NormalizationOps.rmsNorm(empty, gain, EPSILON).sumTensor().backward();
        assertArrayEquals(empty.shape(), empty.grad().shape());
    }

    private static Tensor input() {
        return Tensor.from(new float[]{0.4f, -0.2f, 0.7f, 0.1f, 0.6f, -0.3f}, 2, 1, 3);
    }
    private static Tensor gain() { return Tensor.from(new float[]{0.5f, 1.2f, -0.4f}); }
    private static Tensor upstream() {
        return Tensor.from(new float[]{0.2f, -0.3f, 0.1f, 0.4f, 0.4f, 0}, 2, 1, 3);
    }
}
