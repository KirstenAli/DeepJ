package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.InferenceNormalizationBackend;
import io.github.kirstenali.deepj.tensor.Tensor;

/** Reuses output-only RMS kernels during captured model inference. */
final class ModelRms {
    private ModelRms() {}

    static boolean supports(Tensor input, Tensor gain, float epsilon) {
        return ModelInference.active() && compactInference(input, epsilon) && parameterShape(input, gain)
                && Tensor.backend() instanceof InferenceNormalizationBackend;
    }

    private static boolean compactInference(Tensor input, float epsilon) {
        return !GradMode.isEnabled() && eligibleLayout(input) && input.rank() > 0 && input.numel() > 0
                && Float.isFinite(epsilon) && epsilon > 0;
    }

    private static boolean eligibleLayout(Tensor input) {
        return input.dtype() != DType.FLOAT32 || channelLayout(input);
    }

    private static boolean channelLayout(Tensor input) {
        return input.rank() == 4 && input.strides()[3] != 1;
    }

    private static boolean parameterShape(Tensor input, Tensor gain) {
        return gain != null && gain.rank() > 0 && gain.rank() <= input.rank()
                && gain.size(-1) == input.size(-1) && gain.numel() == input.size(-1);
    }

    static Tensor inference(Tensor input, Tensor gain, Tensor bias, float epsilon, boolean gemma) {
        var backend = (InferenceNormalizationBackend) Tensor.backend();
        if (!gemma && gain.dtype() != DType.FLOAT32) return backend.storedRmsForInference(input, gain, bias, epsilon);
        if (channelLayout(input)) {
            Tensor scale = gain.to(DType.FLOAT32);
            if (gemma) scale = scale.addScalar(1);
            return backend.composedRmsForInference(input, scale, bias, epsilon);
        }
        if (bias == null) return backend.affineRmsForInference(input, gain, epsilon, gemma);
        Tensor scale = gain.to(DType.FLOAT32);
        if (gemma) scale = scale.addScalar(1);
        Tensor result = backend.normalizeForInference(input, scale, null, epsilon, false);
        return result.add(bias).to(input.dtype());
    }
}
