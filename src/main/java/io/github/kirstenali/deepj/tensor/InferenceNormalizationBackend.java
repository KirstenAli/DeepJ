package io.github.kirstenali.deepj.tensor;

/** Optional output-only normalization for generation without a backward graph. */
public interface InferenceNormalizationBackend extends TensorBackend {
    Tensor normalizeForInference(Tensor input, Tensor gamma, Tensor beta, float epsilon, boolean centered);

    /** FP32 RMS statistics and affine arithmetic, with one final conversion to the input dtype. */
    default Tensor affineRmsForInference(Tensor input, Tensor gamma, float epsilon, boolean unitOffset) {
        Tensor scale = gamma.to(DType.FLOAT32);
        if (unitOffset) scale = scale.addScalar(1);
        return normalizeForInference(input, scale, null, epsilon, false).to(input.dtype());
    }
}
