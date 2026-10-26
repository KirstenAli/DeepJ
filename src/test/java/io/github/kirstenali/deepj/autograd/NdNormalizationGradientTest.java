package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertTensorGradients;
import static org.junit.jupiter.api.Assertions.*;

class NdNormalizationGradientTest {
    private static final float EPSILON = 1e-5f;
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void layerNormalizationInputAndParametersMatchFiniteDifferences() {
        Tensor input = inputs(), gamma = gains(), beta = Tensor.from(new float[]{0.1f, 0.2f, -0.3f}, 1, 3);
        assertTensorGradients(input, value -> NormalizationOps.layerNorm(value, gamma, beta, EPSILON),
                upstream(), 1e-3f, 2e-4f, 2e-3f);
        assertTensorGradients(gamma, value -> NormalizationOps.layerNorm(input, value, beta, EPSILON),
                upstream(), 1e-3f, 2e-4f, 2e-3f);
        assertTensorGradients(beta, value -> NormalizationOps.layerNorm(input, gamma, value, EPSILON),
                upstream(), 1e-3f, 2e-4f, 2e-3f);
    }

    @Test
    void rmsInputAndVectorGainsMatchFiniteDifferences() {
        Tensor input = inputs(), gamma = gains().reshape(3);
        assertTensorGradients(input, value -> NormalizationOps.rmsNorm(value, gamma, EPSILON),
                upstream(), 1e-3f, 2e-4f, 2e-3f);
        assertTensorGradients(gamma, value -> NormalizationOps.rmsNorm(input, value, EPSILON),
                upstream(), 1e-3f, 2e-4f, 2e-3f);
        Tensor vector = input.reshape(2, 3).select(0, 0);
        assertTensorGradients(vector, value -> NormalizationOps.rmsNorm(value, gains(), EPSILON),
                Tensor.from(new float[]{0.2f, 0.1f, 0.3f}), 1e-3f, 2e-4f, 2e-3f);
    }

    @Test
    void compactLeavesUseFloat32SavedStatisticsAndGradients() {
        for (DType dtype : new DType[]{DType.FLOAT16, DType.BFLOAT16}) {
            Tensor input = inputs().to(dtype).requiresGrad(true), gamma = gains().to(dtype).requiresGrad(true);
            Tensor reference = Tensor.from(input.toFloatArray(), input.shape()).requiresGrad(true);
            Tensor referenceGain = Tensor.from(gamma.toFloatArray(), gamma.shape()).requiresGrad(true);
            NormalizationOps.rmsNorm(input, gamma, EPSILON).backward(upstream());
            NormalizationOps.rmsNorm(reference, referenceGain, EPSILON).backward(upstream());
            assertArrayEquals(reference.grad().toFloatArray(), input.grad().toFloatArray());
            assertArrayEquals(referenceGain.grad().toFloatArray(), gamma.grad().toFloatArray());
            assertEquals(DType.FLOAT32, input.grad().dtype());
            assertEquals(DType.FLOAT32, gamma.grad().dtype());
        }
    }

    @Test
    void compactLayerNormMatchesDecodedGradientArithmeticIncludingVectorBias() {
        for (DType dtype : new DType[]{DType.FLOAT16, DType.BFLOAT16}) {
            checkCompactLayerNorm(dtype);
        }
    }

    private static void checkCompactLayerNorm(DType dtype) {
        Tensor input = inputs().to(dtype).requiresGrad(true), gamma = gains().to(dtype).requiresGrad(true);
        Tensor beta = Tensor.from(new float[]{0.1f, 0.2f, -0.3f}, dtype).requiresGrad(true);
        Tensor reference = Tensor.from(input.toFloatArray(), input.shape()).requiresGrad(true);
        Tensor gain = Tensor.from(gamma.toFloatArray(), gamma.shape()).requiresGrad(true);
        Tensor bias = Tensor.from(beta.toFloatArray()).requiresGrad(true);
        NormalizationOps.layerNorm(input, gamma, beta, EPSILON).backward(upstream());
        NormalizationOps.layerNorm(reference, gain, bias, EPSILON).backward(upstream());
        assertArrayEquals(reference.grad().toFloatArray(), input.grad().toFloatArray());
        assertArrayEquals(gain.grad().toFloatArray(), gamma.grad().toFloatArray());
        assertArrayEquals(bias.grad().toFloatArray(), beta.grad().toFloatArray());
    }

    @Test
    void vectorInputsReturnMatrixGainGradientsAndMatrixInputsAcceptVectorBias() {
        Tensor input = Tensor.from(new float[]{0.2f, -0.4f, 0.7f}).requiresGrad(true);
        Tensor gamma = gains().requiresGrad(true), beta = new Tensor(3).requiresGrad(true);
        NormalizationOps.layerNorm(input, gamma, beta, EPSILON)
                .backward(Tensor.from(new float[]{0.2f, 0.3f, 0.4f}));
        assertArrayEquals(new int[]{1, 3}, gamma.grad().shape());
        assertArrayEquals(new int[]{3}, beta.grad().shape());
        assertArrayEquals(new int[]{2, 3}, NormalizationOps.layerNorm(
                inputs().reshape(2, 3), gamma, beta, EPSILON).shape());
    }

    @Test
    void directRmsBackwardHandlesNonMatrixSavedStatistics() {
        Tensor input = inputs(), gamma = gains();
        var saved = Tensor.backend().rmsNorm(input, gamma, EPSILON);
        Tensor actual = Tensor.backend().rmsNormBackward(upstream(), saved.normalized(), saved.rms(), gamma);
        input.requiresGrad(true);
        NormalizationOps.rmsNorm(input, gamma, EPSILON).backward(upstream());
        assertArrayEquals(input.grad().toFloatArray(), actual.toFloatArray());
    }

    @Test
    void emptyLeadingAxesHaveZeroGradientsAndSavedValuesRejectMutation() {
        Tensor input = new Tensor(0, 2, 3).requiresGrad(true), gamma = gains().requiresGrad(true);
        NormalizationOps.rmsNorm(input, gamma, EPSILON).sumTensor().backward();
        assertArrayEquals(input.shape(), input.grad().shape());
        assertArrayEquals(new float[]{0, 0, 0}, gamma.grad().toFloatArray());
        Tensor source = inputs().requiresGrad(true);
        Tensor output = NormalizationOps.rmsNorm(source, gamma, EPSILON);
        try (var ignored = GradMode.noGrad()) { gamma.addScalarInPlace(1); }
        assertThrows(IllegalStateException.class, () -> output.backward(upstream()));
    }

    @Test
    void backwardChecksDenominatorAndUpstreamShapes() {
        Tensor input = inputs(), gamma = gains();
        assertThrows(IllegalArgumentException.class, () -> Tensor.backend().normalizationBackward(
                input, input, new Tensor(2), gamma, true));
        assertThrows(IllegalArgumentException.class, () -> Tensor.backend().normalizationBackward(
                new Tensor(1, 2, 3), input, new Tensor(2, 1, 1), gamma, false));
    }

    private static Tensor inputs() { return Tensor.from(new float[]{0.2f, -0.4f, 0.7f, 1.1f, 0.3f, -0.2f}, 2, 1, 3); }
    private static Tensor gains() { return Tensor.from(new float[]{0.7f, 1.1f, -0.3f}, 1, 3); }
    private static Tensor upstream() { return Tensor.from(new float[]{0.3f, 0.2f, -0.5f, 0.4f, 0.1f, -0.2f}, 2, 1, 3); }
}
