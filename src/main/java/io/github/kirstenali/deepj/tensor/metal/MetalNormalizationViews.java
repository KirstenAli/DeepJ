package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.NormalizationChecks;
import io.github.kirstenali.deepj.tensor.NormalizationResult;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;

/** Contiguous FLOAT32 batches reuse the exact established matrix kernels; reshapes only change metadata. */
final class MetalNormalizationViews {
    private MetalNormalizationViews() {}

    static boolean forward(Tensor input, Tensor gamma, Tensor beta, boolean centered) {
        return MetalDenseOps.dense(input) && MetalDenseOps.dense(gamma)
                && (!centered || MetalDenseOps.dense(beta));
    }

    static boolean backward(Tensor gradient, Tensor normalized, Tensor denominator, Tensor gamma) {
        return MetalDenseOps.dense(gradient) && MetalDenseOps.dense(normalized)
                && MetalDenseOps.dense(denominator) && MetalDenseOps.dense(gamma);
    }

    static NormalizationResult normalize(MetalCoreOps backend, Tensor input, Tensor gamma, Tensor beta,
                                          float epsilon, boolean centered) {
        try (var scope = TensorScope.open()) {
            var result = normalizeMatrix(backend, input, gamma, beta, epsilon, centered);
            return new NormalizationResult(scope.escape(result.output()), scope.escape(result.normalized()),
                    scope.escape(result.denominator()));
        }
    }

    private static NormalizationResult normalizeMatrix(MetalCoreOps backend, Tensor input, Tensor gamma, Tensor beta,
                                                        float epsilon, boolean centered) {
        int width = input.size(-1), rows = input.numel() / width;
        Tensor matrix = backend.reshape(input, rows, width), gain = backend.reshape(gamma, 1, width);
        if (centered) {
            return restore(backend, input, layer(backend, matrix, gain, backend.reshape(beta, 1, width), epsilon));
        }
        var result = backend.rmsNorm(matrix, gain, epsilon);
        return restore(backend, input, new NormalizationResult(result.output(), result.normalized(), result.rms()));
    }

    private static NormalizationResult layer(MetalCoreOps backend, Tensor input, Tensor gamma,
                                              Tensor beta, float epsilon) {
        Tensor std = backend.sqrt(backend.addScalar(backend.varianceAlongRows(input), epsilon));
        Tensor normalized = backend.divideBroadcastCols(
                backend.subtractBroadcastCols(input, backend.meanAlongRows(input)), std);
        Tensor output = backend.addBroadcastRows(backend.multiplyBroadcastRows(normalized, gamma), beta);
        return new NormalizationResult(output, normalized, std);
    }

    private static NormalizationResult restore(MetalCoreOps backend, Tensor input, NormalizationResult result) {
        return new NormalizationResult(backend.reshape(result.output(), input.shape()),
                backend.reshape(result.normalized(), input.shape()),
                backend.reshape(result.denominator(), NormalizationChecks.statisticsShape(input)));
    }

    static Tensor gradient(MetalCoreOps backend, Tensor gradient, Tensor normalized, Tensor denominator,
                             Tensor gamma, boolean centered) {
        try (var scope = TensorScope.open()) {
            return scope.escape(gradientMatrix(backend, gradient, normalized, denominator, gamma, centered));
        }
    }

    private static Tensor gradientMatrix(MetalCoreOps backend, Tensor gradient, Tensor normalized, Tensor denominator,
                                          Tensor gamma, boolean centered) {
        int width = normalized.size(-1), rows = normalized.numel() / width;
        Tensor g = backend.reshape(gradient, rows, width), norm = backend.reshape(normalized, rows, width);
        Tensor std = backend.reshape(denominator, rows, 1), gain = backend.reshape(gamma, 1, width);
        Tensor result = centered ? backend.layerNormBackward(backend.multiplyBroadcastRows(g, gain), norm, std, width)
                : backend.rmsNormBackward(g, norm, std, gain);
        return backend.reshape(result, gradient.shape());
    }
}
