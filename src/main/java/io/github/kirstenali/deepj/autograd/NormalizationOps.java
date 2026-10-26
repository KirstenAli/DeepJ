package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;

public final class NormalizationOps {
    private NormalizationOps() {}

    public static Tensor rmsNorm(Tensor input, Tensor gamma, float epsilon) {
        var result = Tensor.backend().rmsNorm(input, gamma, epsilon);
        Tensor normalized = result.normalized(), rms = result.rms();
        boolean dx = input.requiresGrad(), dg = gamma.requiresGrad();
        return Autograd.recordBuiltinSaved(result.output(), g -> new Tensor[]{
                dx ? Tensor.backend().rmsNormBackward(g, normalized, rms, gamma) : null,
                dg ? g.multiply(normalized).sumRows() : null},
                new Tensor[]{input, gamma}, gamma, normalized, rms);
    }

    public static Tensor layerNorm(Tensor input, Tensor gamma, Tensor beta, float epsilon) {
        Tensor standardDeviation = Tensor.backend().varianceAlongRows(input).addScalar(epsilon).sqrt();
        Tensor normalized;
        Tensor output;
        try (var ignored = GradMode.noGrad()) {
            normalized = input.subtractBroadcastCols(input.meanAlongRows()).divideBroadcastCols(standardDeviation);
            output = normalized.multiplyBroadcastRows(gamma).addBroadcastRows(beta);
        }
        var required = new RequiredGradients(input.requiresGrad(), gamma.requiresGrad(), beta.requiresGrad());
        return Autograd.recordBuiltinSaved(output,
                g -> layerGradients(g, gamma, normalized, standardDeviation, required),
                new Tensor[]{input, gamma, beta}, gamma, normalized, standardDeviation);
    }

    private static Tensor[] layerGradients(Tensor g, Tensor gamma, Tensor normalized,
                                            Tensor standardDeviation, RequiredGradients required) {
        return new Tensor[]{required.input() ? Tensor.backend().layerNormBackward(
                g.multiplyBroadcastRows(gamma), normalized, standardDeviation, normalized.matrixCols()) : null,
                required.gain() ? g.multiply(normalized).sumRows() : null,
                required.bias() ? g.sumRows() : null};
    }

    private record RequiredGradients(boolean input, boolean gain, boolean bias) {}
}
