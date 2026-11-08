package io.github.kirstenali.deepj.tensor;

/** Optional output-only normalization for generation without a backward graph. */
public interface InferenceNormalizationBackend extends TensorBackend {
    Tensor normalizeForInference(Tensor input, Tensor gamma, Tensor beta, float epsilon, boolean centered);

    /** Rounds normalized values and affine operations in the parameter storage dtype. */
    default Tensor storedRmsForInference(Tensor input, Tensor gamma, Tensor beta, float epsilon) {
        Tensor value = input.to(DType.FLOAT32);
        Tensor denominator = value.multiply(value).mean(-1, true).addScalar(epsilon).sqrt();
        Tensor result = value.divide(denominator).to(gamma.dtype()).multiply(gamma);
        if (beta != null) result = result.add(beta);
        return result.to(input.dtype());
    }

    /** Preserves the separate FLOAT32 square, mean, sqrt, divide and affine rounding steps. */
    default Tensor composedRmsForInference(Tensor input, Tensor gamma, Tensor beta, float epsilon) {
        Tensor value = input.to(DType.FLOAT32);
        Tensor denominator = value.multiply(value).mean(-1, true).addScalar(epsilon).sqrt();
        Tensor result = value.divide(denominator).multiply(gamma.to(DType.FLOAT32));
        if (beta != null) result = result.add(beta.to(DType.FLOAT32));
        return result.to(input.dtype());
    }

    /** FP32 RMS statistics and affine arithmetic, with one final conversion to the input dtype. */
    default Tensor affineRmsForInference(Tensor input, Tensor gamma, float epsilon, boolean unitOffset) {
        Tensor scale = gamma.to(DType.FLOAT32);
        if (unitOffset) scale = scale.addScalar(1);
        return normalizeForInference(input, scale, null, epsilon, false).to(input.dtype());
    }
}
