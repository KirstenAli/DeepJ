package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.NormalizationChecks;
import io.github.kirstenali.deepj.tensor.NormalizationResult;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.InferenceNormalizationBackend;
import io.github.kirstenali.deepj.autograd.GradMode;

/** General normalization never falls back to CPU numerical evaluation. */
abstract class MetalGeneralNormalizationOps extends MetalInPlaceOps implements InferenceNormalizationBackend {
    @Override
    public Tensor normalizeForInference(Tensor input, Tensor gamma, Tensor beta, float epsilon, boolean centered) {
        if (GradMode.isEnabled()) throw new IllegalStateException("Inference normalization requires noGrad");
        return graph.layoutNormalizeInference(input, gamma, beta, epsilon, centered);
    }

    @Override
    public NormalizationResult normalize(Tensor input, Tensor gamma, Tensor beta, float epsilon, boolean centered) {
        NormalizationChecks.forward(input, gamma, beta, epsilon, centered);
        return MetalNormalizationViews.forward(input, gamma, beta, centered)
                ? MetalNormalizationViews.normalize(this, input, gamma, beta, epsilon, centered)
                : graph.layoutNormalize(input, gamma, beta, epsilon, centered);
    }

    @Override
    public Tensor normalizationBackward(Tensor gradient, Tensor normalized, Tensor denominator,
                                         Tensor gamma, boolean centered) {
        NormalizationChecks.backward(gradient, normalized, denominator, gamma);
        return MetalNormalizationViews.backward(gradient, normalized, denominator, gamma)
                ? MetalNormalizationViews.gradient(this, gradient, normalized, denominator, gamma, centered)
                : graph.layoutNormalizationBackward(gradient, normalized, denominator, gamma, centered);
    }
}
