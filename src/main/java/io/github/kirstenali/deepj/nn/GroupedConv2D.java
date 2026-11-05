package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.TensorScope;
import io.github.kirstenali.deepj.tensor.BackwardBuffers;
import java.util.ArrayList;
import java.util.List;

/** Same-padded grouped cross-correlation, using existing patch gathering and backend matrix multiplication. */
public final class GroupedConv2D {
    private static final long TILE_ELEMENTS = 8L * 1024 * 1024;

    private GroupedConv2D() {}

    public static Tensor apply(Tensor input, Tensor weight, Tensor bias, int groups) {
        validate(input, weight, bias, groups);
        try (var scope = TensorScope.open()) {
            Tensor result = weight.size(2) == 1 && weight.size(3) == 1
                    ? pointwise(input, weight, groups) : tiled(input, weight, groups);
            if (bias != null) result = result.add(bias.reshape(1, weight.size(0), 1, 1));
            return scope.escape(result);
        }
    }

    private static void validate(Tensor input, Tensor weight, Tensor bias, int groups) {
        NeuralChecks.rank(input, 4);
        NeuralChecks.rank(weight, 4);
        if (groups < 1 || input.size(1) != weight.size(1) * groups || weight.size(0) % groups != 0) {
            throw new IllegalArgumentException("Invalid convolution groups or channels");
        }
        if (weight.size(2) % 2 != 1 || weight.size(3) % 2 != 1) {
            throw new IllegalArgumentException("Same padding requires odd kernels");
        }
        validateBias(weight, bias);
    }

    private static void validateBias(Tensor weight, Tensor bias) {
        if (bias != null && (bias.rank() != 1 || bias.size(0) != weight.size(0))) {
            throw new IllegalArgumentException("Invalid convolution bias");
        }
    }

    private static Tensor pointwise(Tensor input, Tensor weight, int groups) {
        Tensor rows = input.reshape(input.size(0), groups, weight.size(1), input.size(2) * input.size(3))
                .permute(0, 1, 3, 2);
        return product(rows, weight, groups, input.size(2), input.size(3));
    }

    private static Tensor tiled(Tensor input, Tensor weight, int groups) {
        int rows = tileRows(input, weight);
        List<Tensor> tiles = new ArrayList<>();
        for (int start = 0; start < input.size(2); start += rows) {
            if (Thread.currentThread().isInterrupted()) {
                throw new java.util.concurrent.CancellationException("Convolution stopped");
            }
            tiles.add(evaluateTile(input, weight, groups, start, Math.min(input.size(2), start + rows)));
        }
        return TensorJoin.concatenate(2, tiles.toArray(Tensor[]::new));
    }

    private static Tensor evaluateTile(Tensor input, Tensor weight, int groups, int start, int end) {
        if (GradMode.isEnabled()) return tile(input, weight, groups, start, end);
        try (var buffers = BackwardBuffers.begin()) {
            Tensor result = buffers.capture(() -> tile(input, weight, groups, start, end));
            buffers.expose(result);
            result.materialize();
            buffers.releaseTemporaryResources();
            return result;
        }
    }

    private static int tileRows(Tensor input, Tensor weight) {
        long perRow = (long) input.size(0) * input.size(1) * input.size(3) * weight.size(2) * weight.size(3);
        return (int) Math.max(1, Math.min(input.size(2), TILE_ELEMENTS / perRow));
    }

    private static Tensor tile(Tensor input, Tensor weight, int groups, int start, int end) {
        try (var scope = TensorScope.open()) {
            int padding = weight.size(2) / 2;
            int first = Math.max(0, start - padding), last = Math.min(input.size(2), end + padding);
            var window = new Window2D(weight.size(2), weight.size(3), 1, 1,
                    padding, weight.size(3) / 2, 1, 1);
            Tensor patches = SpatialPatches.gather(input.slice(2, first, last), window, 0).values();
            Tensor selected = patches.reshape(input.size(0), input.size(1), last - first, input.size(3), window.area())
                    .slice(2, start - first, end - first);
            Tensor rows = selected.reshape(input.size(0), groups, weight.size(1), end - start,
                    input.size(3), window.area()).permute(0, 1, 3, 4, 2, 5)
                    .reshape(input.size(0), groups, (end - start) * input.size(3), weight.size(1) * window.area());
            return scope.escape(product(rows, weight, groups, end - start, input.size(3)));
        }
    }

    private static Tensor product(Tensor rows, Tensor weight, int groups, int height, int width) {
        Tensor matrix = weight.reshape(groups, weight.size(0) / groups, rows.size(-1)).transpose(-1, -2);
        return rows.matmul(matrix).permute(0, 1, 3, 2)
                .reshape(rows.size(0), weight.size(0), height, width);
    }
}
