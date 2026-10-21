package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class NormalizationGradientTest {
    @BeforeEach
    void useCpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void rmsNormCanDifferentiateTheInputOrGainIndependently() {
        Tensor input = input(), gain = Tensor.from2D(new float[][]{{0.8f, 1.2f, 0.9f}});
        OperationGradientAssertions.check(x -> NormalizationOps.rmsNorm(x, gain, 1e-6f), input);
        OperationGradientAssertions.check(g -> NormalizationOps.rmsNorm(input, g, 1e-6f), gain);
    }

    @Test
    void layerNormSupportsFrozenInputsGainAndBias() {
        Tensor input = input(), gain = Tensor.ones(1, 3), bias = Tensor.zeros(1, 3);
        OperationGradientAssertions.check(x -> NormalizationOps.layerNorm(x, gain, bias, 1e-5f), input);
        OperationGradientAssertions.check(g -> NormalizationOps.layerNorm(input, g, bias, 1e-5f), gain);
        OperationGradientAssertions.check(b -> NormalizationOps.layerNorm(input, gain, b, 1e-5f), bias);
    }

    private static Tensor input() {
        return Tensor.from2D(new float[][]{{0.2f, -0.7f, 1.3f}, {-0.4f, 0.8f, 1.5f}});
    }
}
