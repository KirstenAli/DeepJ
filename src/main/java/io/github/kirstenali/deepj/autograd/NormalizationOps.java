package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;

public final class NormalizationOps {
    private NormalizationOps() {}

    public static Tensor rmsNorm(Tensor input, Tensor gamma, float epsilon) {
        var result = Tensor.backend().rmsNorm(input, gamma, epsilon);
        return Autograd.recordSaved(result.output(), g -> new Tensor[]{
                input.requiresGrad() ? Tensor.backend().rmsNormBackward(g,
                        result.normalized(), result.rms(), gamma) : null,
                gamma.requiresGrad() ? g.multiply(result.normalized()).sumRows() : null},
                new Tensor[]{input, gamma}, input, gamma, result.normalized(), result.rms());
    }

    public static Tensor layerNorm(Tensor input, Tensor gamma, Tensor beta, float epsilon) {
        Tensor standardDeviation = Tensor.backend().varianceAlongRows(input).addScalar(epsilon).sqrt();
        Tensor normalized;
        Tensor output;
        try (var ignored = GradMode.noGrad()) {
            normalized = input.subtractBroadcastCols(input.meanAlongRows()).divideBroadcastCols(standardDeviation);
            output = normalized.multiplyBroadcastRows(gamma).addBroadcastRows(beta);
        }
        return Autograd.recordSaved(output, g -> layerGradients(g, input, gamma, beta, normalized, standardDeviation),
                new Tensor[]{input, gamma, beta}, input, gamma, beta, normalized, standardDeviation);
    }

    private static Tensor[] layerGradients(Tensor g, Tensor input, Tensor gamma, Tensor beta,
                                            Tensor normalized, Tensor standardDeviation) {
        return new Tensor[]{input.requiresGrad() ? Tensor.backend().layerNormBackward(
                g.multiplyBroadcastRows(gamma), normalized, standardDeviation, input.cols) : null,
                gamma.requiresGrad() ? g.multiply(normalized).sumRows() : null,
                beta.requiresGrad() ? g.sumRows() : null};
    }
}
