package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;

/** Compact normalization uses bounded row scratch and the same FLOAT32 statistics as the scalar path. */
final class CpuCompactNormalization {
    private CpuCompactNormalization() {}

    static boolean forward(Tensor input, Tensor gamma, Tensor beta, NormalizationResult result,
                             float epsilon, boolean centered) {
        Tensor bias = centered ? beta : gamma;
        if (!supported(input, gamma, bias) || allFloat(input, gamma, bias)) return false;
        CpuBlas.compactNormalize(storage(input), storage(gamma), storage(bias), storage(result.output()),
                result.normalized().memory().floats(), result.denominator().memory().floats(),
                new int[]{input.storageOffset(), gamma.storageOffset(), bias.storageOffset(),
                    input.numel() / input.size(-1), input.size(-1), centered ? 1 : 0, workers(input),
                    code(input), code(gamma), code(bias), code(result.output())}, epsilon);
        return true;
    }

    static boolean backward(Tensor gradient, Tensor normalized, Tensor denominator, Tensor gamma,
                              Tensor output, boolean centered) {
        if (!supported(gradient, gamma, normalized) || !floatStatistics(normalized, denominator)
                || allFloat(gradient, gamma, normalized)) return false;
        CpuBlas.compactNormalizationBackward(storage(gradient), normalized.memory().floats(),
                denominator.memory().floats(), storage(gamma), output.memory().floats(),
                new int[]{gradient.storageOffset(), normalized.storageOffset(), denominator.storageOffset(),
                    gamma.storageOffset(), gradient.numel() / gradient.size(-1), gradient.size(-1),
                    centered ? 1 : 0, workers(gradient), code(gradient), code(gamma)});
        return true;
    }

    private static boolean supported(Tensor input, Tensor gamma, Tensor beta) {
        return CpuBlas.AVAILABLE && input.numel() >= 256 && input.size(-1) <= 8192
                && input.isContiguous() && gamma.isContiguous() && beta.isContiguous();
    }

    private static boolean allFloat(Tensor input, Tensor gamma, Tensor beta) {
        return input.dtype() == DType.FLOAT32 && gamma.dtype() == DType.FLOAT32 && beta.dtype() == DType.FLOAT32;
    }

    private static boolean floatStatistics(Tensor normalized, Tensor denominator) {
        return normalized.dtype() == DType.FLOAT32 && denominator.dtype() == DType.FLOAT32
                && denominator.isContiguous();
    }

    private static Object storage(Tensor tensor) { return new CpuProductValues(tensor).storage(); }
    private static int code(Tensor tensor) { return tensor.dtype().storageCode(); }

    private static int workers(Tensor input) {
        if (!DeepJExecutor.isParallelEnabled()) return 1;
        return Math.max(1, Math.min(DeepJExecutor.getNumThreads(), input.numel() / 32768));
    }
}
