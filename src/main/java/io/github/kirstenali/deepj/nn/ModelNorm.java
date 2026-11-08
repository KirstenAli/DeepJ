package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.InferenceNormalizationBackend;
import io.github.kirstenali.deepj.tensor.Tensor;

/** Uses the backend's output-only centered normalization during bounded model inference. */
final class ModelNorm {
    private ModelNorm() {}

    static boolean supports(Tensor input, float epsilon) {
        return !GradMode.isEnabled() && ModelInference.active() && input.rank() > 0 && input.numel() > 0
                && Float.isFinite(epsilon) && epsilon > 0
                && Tensor.backend() instanceof InferenceNormalizationBackend;
    }

    static Tensor inference(Tensor input, float epsilon) {
        Tensor gain = Tensor.ones(input.dtype(), input.size(-1)), bias = new Tensor(input.dtype(), input.size(-1));
        return ((InferenceNormalizationBackend) Tensor.backend())
                .normalizeForInference(input, gain, bias, epsilon, true);
    }
}
