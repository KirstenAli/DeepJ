// Portions adapted from Diffusers (Apache-2.0); see META-INF/licenses/sana-video-NOTICE.txt.
package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.nn.ModelMemory;
import io.github.kirstenali.deepj.nn.ModelOps;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;

final class WanDecoderBlocks {
    private final TensorWeights weights;
    private final WanFeatureCache cache;

    WanDecoderBlocks(TensorWeights weights, WanFeatureCache cache) {
        this.weights = weights;
        this.cache = cache;
    }

    Tensor norm(Tensor input, String name) {
        Tensor length = input.multiply(input).sum(1, true).sqrt().clamp(1e-12f, Float.MAX_VALUE);
        Tensor result = input.divide(length).multiplyScalar((float) Math.sqrt(input.size(1)))
                .multiply(weights.tensor(name + ".gamma"));
        return weights.contains(name + ".bias") ? result.add(weights.tensor(name + ".bias")) : result;
    }

    Tensor activation(Tensor input, String name) {
        try (var scope = TensorScope.open()) {
            return scope.escape(VideoMemory.finish(ModelOps.silu(norm(input, name))));
        }
    }

    Tensor residual(Tensor input, String name) {
        try (var scope = TensorScope.open(); var pinned = ModelMemory.retain(input)) {
            ModelMemory.checkCancelled();
            Tensor first = cache.convolution(activation(input, name + ".norm1"), name + ".conv1");
            Tensor second = cache.convolution(activation(first, name + ".norm2"), name + ".conv2");
            try (var fixed = ModelMemory.retain(second)) {
                Tensor shortcut = weights.contains(name + ".conv_shortcut.weight")
                        ? cache.convolution(input, name + ".conv_shortcut") : input;
                return scope.escape(VideoMemory.finish(second.add(shortcut)));
            }
        }
    }

    Tensor attention(Tensor input, String name) {
        try (var scope = TensorScope.open(); var pinned = ModelMemory.retain(input)) {
            Tensor frames = VideoLayout.frames(input);
            Tensor projected = VideoConvolution.spatial(norm(frames, name + ".norm"), weights, name + ".to_qkv", 1);
            int channels = input.size(1);
            Tensor query = heads(projected, 0, channels), key = heads(projected, channels, 2 * channels);
            Tensor value = heads(projected, 2 * channels, 3 * channels);
            Tensor attended = ModelOps.attention(query, key, value, null, (float) (1 / Math.sqrt(channels)), 0);
            Tensor image = ModelOps.image(ModelOps.mergeHeads(attended), input.size(3), input.size(4));
            image = VideoConvolution.spatial(image, weights, name + ".proj", 1);
            return scope.escape(VideoMemory.finish(input.add(VideoLayout.video(image, input.size(0), input.size(2)))));
        }
    }

    private static Tensor heads(Tensor qkv, int begin, int end) {
        return ModelOps.heads(ModelOps.imageTokens(qkv.slice(1, begin, end)), 1);
    }

    Tensor upsample(Tensor input, String name, boolean temporal) {
        Tensor expanded = temporal ? cache.upsampleTime(input, name + ".time_conv") : input;
        Tensor frames = VideoLayout.frames(expanded);
        frames = VideoConvolution.spatial(ModelOps.upsample(frames), weights, name + ".resample.1", 1);
        return VideoLayout.video(frames, expanded.size(0), expanded.size(2));
    }
}
