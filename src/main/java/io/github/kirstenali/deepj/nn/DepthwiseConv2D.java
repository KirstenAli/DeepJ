package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.BackwardBuffers;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.util.ArrayList;
import java.util.List;

/** Same-padded depthwise cross-correlation using shared spatial patches and tensor reductions. */
public final class DepthwiseConv2D {
    private static final long TILE_ELEMENTS = 8L * 1024 * 1024;
    private DepthwiseConv2D() {}

    public static Tensor apply(Tensor input, Tensor weight, Tensor bias) {
        validate(input, weight, bias);
        try (var scope = TensorScope.open()) {
            Tensor result = ConvolutionExecution.inference(input, weight, input.size(1));
            if (result == null) result = composed(input, weight);
            if (bias != null) result = result.add(bias.reshape(1, input.size(1), 1, 1));
            return scope.escape(result);
        }
    }

    private static Tensor composed(Tensor input, Tensor weight) {
        int channels = tileChannels(input, weight);
        List<Tensor> output = new ArrayList<>();
        try (var scope = TensorScope.open()) {
            for (int start = 0; start < input.size(1); start += channels) {
                ModelMemory.checkCancelled();
                int end = Math.min(input.size(1), start + channels);
                output.add(evaluate(input.slice(1, start, end), weight.slice(0, start, end)));
            }
            return scope.escape(TensorJoin.concatenate(1, output.toArray(Tensor[]::new)));
        }
    }

    private static void validate(Tensor input, Tensor weight, Tensor bias) {
        NeuralChecks.rank(input, 4);
        NeuralChecks.rank(weight, 4);
        if (weight.size(0) != input.size(1) || weight.size(1) != 1
                || weight.size(2) % 2 != 1 || weight.size(3) % 2 != 1) {
            throw new IllegalArgumentException("Expected one odd spatial kernel per input channel");
        }
        if (bias != null && (bias.rank() != 1 || bias.size(0) != input.size(1))) {
            throw new IllegalArgumentException("Expected one bias per depthwise channel");
        }
    }

    private static int tileChannels(Tensor input, Tensor weight) {
        long elements = (long) input.size(0) * input.size(2) * input.size(3) * weight.size(2) * weight.size(3);
        return (int) Math.max(1, Math.min(input.size(1), TILE_ELEMENTS / elements));
    }

    private static Tensor evaluate(Tensor input, Tensor weight) {
        if (GradMode.isEnabled()) return tile(input, weight);
        try (var buffers = BackwardBuffers.begin()) {
            Tensor result = buffers.capture(() -> tile(input, weight));
            buffers.expose(result);
            result.materialize();
            buffers.releaseTemporaryResources();
            return result;
        }
    }

    private static Tensor tile(Tensor input, Tensor weight) {
        try (var scope = TensorScope.open()) {
            var window = new Window2D(weight.size(2), weight.size(3), 1, 1,
                    weight.size(2) / 2, weight.size(3) / 2, 1, 1);
            Tensor patches = SpatialPatches.gather(input, window, 0).values();
            Tensor kernel = weight.reshape(1, input.size(1), 1, window.area());
            Tensor result = patches.multiply(kernel).sum(-1, false);
            return scope.escape(result.reshape(input.shape()));
        }
    }
}
