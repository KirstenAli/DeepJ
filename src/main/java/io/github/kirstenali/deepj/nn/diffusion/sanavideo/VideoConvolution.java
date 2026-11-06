package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.nn.GroupedConv2D;
import io.github.kirstenali.deepj.nn.DepthwiseConv2D;
import io.github.kirstenali.deepj.nn.ModelOps;
import io.github.kirstenali.deepj.nn.ModelMemory;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;

/** Temporal convolution composes DeepJ views, concatenation, and existing spatial GPU convolution. */
final class VideoConvolution {
    private VideoConvolution() {}

    static Tensor spatial(Tensor input, TensorWeights weights, String name, int groups) {
        Tensor bias = weights.contains(name + ".bias") ? weights.tensor(name + ".bias") : null;
        Tensor kernel = weights.tensor(name + ".weight");
        return groups == input.size(1) && kernel.size(0) == groups
                ? DepthwiseConv2D.apply(input, kernel, bias) : GroupedConv2D.apply(input, kernel, bias, groups);
    }

    static Tensor causal(Tensor input, Tensor weight, Tensor bias, Tensor history) {
        validate(input, weight, history);
        try (var scope = TensorScope.open();
             var pinned = ModelMemory.retain(input, weight, history == null ? input : history)) {
            Tensor result = plane(window(input, history, 1 - weight.size(2)), weight, 0);
            for (int tap = 1; tap < weight.size(2); tap++) {
                Tensor current = window(input, history, tap + 1 - weight.size(2));
                result = VideoMemory.finish(result.add(plane(current, weight, tap)));
            }
            if (bias != null) result = result.add(bias.reshape(1, weight.size(0), 1, 1, 1));
            return scope.escape(result);
        }
    }

    private static Tensor plane(Tensor input, Tensor weight, int tap) {
        Tensor frames = VideoLayout.frames(input);
        Tensor kernel = weight.slice(2, tap, tap + 1)
                .reshape(weight.size(0), weight.size(1), weight.size(3), weight.size(4));
        return VideoLayout.video(GroupedConv2D.apply(frames, kernel, null, 1), input.size(0), input.size(2));
    }

    private static Tensor window(Tensor input, Tensor history, int start) {
        int end = start + input.size(2), available = history == null ? 0 : history.size(2);
        var parts = new java.util.ArrayList<Tensor>();
        int missing = Math.max(0, Math.min(end, -available) - start);
        if (missing > 0) {
            parts.add(Tensor.backend().zeros(input.dtype(), input.size(0), input.size(1), missing,
                    input.size(3), input.size(4)));
        }
        if (history != null && start < 0 && end > -available) {
            parts.add(history.slice(2, Math.max(0, start + available), Math.min(available, end + available)));
        }
        if (end > 0) parts.add(input.slice(2, Math.max(0, start), end));
        return parts.size() == 1 ? parts.get(0) : ModelOps.cat(2, parts.toArray(Tensor[]::new));
    }

    private static void validate(Tensor input, Tensor weight, Tensor history) {
        if (input.rank() != 5 || weight.rank() != 5 || input.size(1) != weight.size(1)) {
            throw new IllegalArgumentException("Expected compatible 5D video and convolution weights");
        }
        if (history != null && history.size(2) > weight.size(2) - 1) {
            throw new IllegalArgumentException("Causal convolution history exceeds the kernel");
        }
    }

    static Tensor patch(Tensor input, Tensor weight, Tensor bias) {
        int pt = weight.size(2), ph = weight.size(3), pw = weight.size(4);
        int t = input.size(2) / pt, h = input.size(3) / ph, w = input.size(4) / pw;
        Tensor patches = input.reshape(input.size(0), input.size(1), t, pt, h, ph, w, pw)
                .permute(0, 2, 4, 6, 1, 3, 5, 7)
                .reshape(input.size(0), t * h * w, input.size(1) * pt * ph * pw);
        return patches.matmul(weight.reshape(weight.size(0), patches.size(-1)).transpose(-1, -2)).add(bias);
    }
}
