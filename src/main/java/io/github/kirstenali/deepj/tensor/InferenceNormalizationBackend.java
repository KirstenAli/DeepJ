package io.github.kirstenali.deepj.tensor;

/** Optional output-only normalization for generation without a backward graph. */
public interface InferenceNormalizationBackend extends TensorBackend {
    Tensor normalizeForInference(Tensor input, Tensor gamma, Tensor beta, float epsilon, boolean centered);
}
