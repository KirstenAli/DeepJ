package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.InferenceNormalizationBackend;
import io.github.kirstenali.deepj.tensor.Tensor;

/** Keeps compact model normalization in FP32 while reusing the backend's output-only kernel. */
final class ModelRms {
    private ModelRms() {}

    static boolean supports(Tensor input, Tensor gain, float epsilon) {
        return ModelInference.active() && compactInference(input, epsilon) && parameterShape(input, gain)
                && Tensor.backend() instanceof InferenceNormalizationBackend;
    }

    private static boolean compactInference(Tensor input, float epsilon) {
        return !GradMode.isEnabled() && input.dtype() != DType.FLOAT32 && input.rank() > 0 && input.numel() > 0
                && Float.isFinite(epsilon) && epsilon > 0;
    }

    private static boolean parameterShape(Tensor input, Tensor gain) {
        return gain != null && gain.rank() > 0 && gain.rank() <= input.rank()
                && gain.size(-1) == input.size(-1) && gain.numel() == input.size(-1);
    }

    static Tensor inference(Tensor input, Tensor gain, Tensor bias, float epsilon, boolean gemma) {
        var backend = (InferenceNormalizationBackend) Tensor.backend();
        if (bias == null) return backend.affineRmsForInference(input, gain, epsilon, gemma);
        Tensor scale = gain.to(DType.FLOAT32);
        if (gemma) scale = scale.addScalar(1);
        Tensor result = backend.normalizeForInference(input, scale, null, epsilon, false);
        return result.add(bias).to(input.dtype());
    }
}
