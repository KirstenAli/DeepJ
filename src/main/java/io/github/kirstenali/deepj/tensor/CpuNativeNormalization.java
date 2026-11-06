package io.github.kirstenali.deepj.tensor;

/** Dense FLOAT32 normalization keeps scalar reduction order in the native CPU implementation. */
final class CpuNativeNormalization {
    private CpuNativeNormalization() {}

    static boolean forward(Tensor input, Tensor gamma, Tensor beta, NormalizationResult result,
                             float epsilon, boolean centered) {
        Tensor bias = centered ? beta : gamma;
        if (!supported(input, gamma, bias) || result.output().dtype() != DType.FLOAT32) return false;
        int width = input.size(-1);
        CpuBlas.normalize(input.memory().floats(), gamma.memory().floats(), bias.memory().floats(),
                result.output().memory().floats(), result.normalized().memory().floats(),
                result.denominator().memory().floats(),
                new int[]{input.storageOffset(), gamma.storageOffset(), bias.storageOffset(),
                    input.numel() / width, width, centered ? 1 : 0}, epsilon);
        return true;
    }

    static boolean backward(Tensor gradient, Tensor normalized, Tensor denominator, Tensor gamma,
                              Tensor output, boolean centered) {
        if (!supported(gradient, normalized, denominator) || !dense(gamma)) return false;
        int width = gradient.size(-1);
        CpuBlas.normalizationBackward(gradient.memory().floats(), normalized.memory().floats(),
                denominator.memory().floats(), gamma.memory().floats(), output.memory().floats(),
                new int[]{gradient.storageOffset(), normalized.storageOffset(), denominator.storageOffset(),
                    gamma.storageOffset(), gradient.numel() / width, width, centered ? 1 : 0});
        return true;
    }

    private static boolean supported(Tensor input, Tensor gamma, Tensor beta) {
        return CpuBlas.AVAILABLE && input.numel() >= 256 && dense(input) && dense(gamma) && dense(beta);
    }

    private static boolean dense(Tensor tensor) {
        return tensor.dtype() == DType.FLOAT32 && tensor.isContiguous();
    }
}
