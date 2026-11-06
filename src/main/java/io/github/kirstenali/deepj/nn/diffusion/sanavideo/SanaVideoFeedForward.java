// Portions adapted from Diffusers (Apache-2.0); see META-INF/licenses/sana-video-NOTICE.txt.
package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.nn.ModelMemory;
import io.github.kirstenali.deepj.nn.ModelOps;
import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;

/** Spatial gated convolution followed by residual temporal mixing. */
final class SanaVideoFeedForward {
    private SanaVideoFeedForward() {}

    static Tensor apply(Tensor input, TensorWeights weights, String name) {
        try (var scope = TensorScope.open(); var pinned = ModelMemory.retain(input)) {
            Tensor frames = VideoLayout.frames(input);
            Tensor spatial = !GradMode.isEnabled() && input.size(2) > 4
                    ? framewise(frames, weights, name) : spatial(frames, weights, name);
            Tensor temporal = VideoLayout.video(spatial, input.size(0), input.size(2))
                    .reshape(input.size(0), input.size(1), input.size(2), input.size(3) * input.size(4));
            temporal = temporal.add(VideoConvolution.spatial(temporal, weights, name + ".conv_temp", 1));
            return scope.escape(temporal.reshape(input.shape()));
        }
    }

    private static Tensor spatial(Tensor input, TensorWeights weights, String name) {
        Tensor hidden = VideoMemory.finish(ModelOps.silu(
                VideoConvolution.spatial(input, weights, name + ".conv_inverted", 1)));
        hidden = VideoConvolution.spatial(hidden, weights, name + ".conv_depth", hidden.size(1));
        int half = hidden.size(1) / 2;
        hidden = VideoMemory.finish(hidden.slice(1, 0, half)
                .multiply(ModelOps.silu(hidden.slice(1, half, 2 * half))));
        return VideoConvolution.spatial(hidden, weights, name + ".conv_point", 1);
    }

    private static Tensor framewise(Tensor input, TensorWeights weights, String name) {
        var results = new java.util.ArrayList<Tensor>();
        var retained = new java.util.ArrayList<ModelMemory.Pin>();
        try (var scope = TensorScope.open(); var fixed = ModelMemory.retain(input)) {
            for (int frame = 0; frame < input.size(0); frame++) {
                ModelMemory.checkCancelled();
                Tensor output = VideoMemory.finish(frame(input.slice(0, frame, frame + 1), weights, name));
                retained.add(ModelMemory.retain(output));
                results.add(output);
            }
            return scope.escape(VideoMemory.finish(ModelOps.cat(0, results.toArray(Tensor[]::new))));
        } finally { retained.forEach(ModelMemory.Pin::close); }
    }

    private static Tensor frame(Tensor input, TensorWeights weights, String name) {
        try (var scope = TensorScope.open()) { return scope.escape(spatial(input, weights, name)); }
    }
}
