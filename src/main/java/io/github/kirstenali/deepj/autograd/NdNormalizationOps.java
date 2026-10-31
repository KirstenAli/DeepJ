package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.NormalizationResult;
import io.github.kirstenali.deepj.tensor.Tensor;

final class NdNormalizationOps {
    private NdNormalizationOps() {}

    static Tensor apply(Tensor input, Tensor gamma, Tensor beta, float epsilon, boolean centered) {
        var result = Tensor.backend().normalize(input, gamma, beta, epsilon, centered);
        boolean dx = input.requiresGrad(), dg = gamma.requiresGrad(), db = centered && beta.requiresGrad();
        Tensor[] parents = centered ? new Tensor[]{input, gamma, beta} : new Tensor[]{input, gamma};
        return Autograd.recordBuiltinSaved(result.output(), gradient -> gradients(gradient, result, gamma,
                beta, centered, dx, dg, db), parents, gamma, result.normalized(), result.denominator());
    }

    private static Tensor[] gradients(Tensor gradient, NormalizationResult result, Tensor gamma, Tensor beta,
                                        boolean centered, boolean dx, boolean dg, boolean db) {
        Tensor input = dx ? Tensor.backend().normalizationBackward(gradient, result.normalized(),
                result.denominator(), gamma, centered) : null;
        Tensor gain = dg ? parameterGradient(gradient.multiply(result.normalized()), gamma) : null;
        if (!centered) return new Tensor[]{input, gain};
        return new Tensor[]{input, gain, db ? parameterGradient(gradient, beta) : null};
    }

    private static Tensor parameterGradient(Tensor gradient, Tensor parameter) {
        if (gradient.rank() < parameter.rank()) return Tensor.backend().reshape(gradient, parameter.shape());
        return Tensor.backend().sumToShape(gradient, parameter.shape());
    }
}
