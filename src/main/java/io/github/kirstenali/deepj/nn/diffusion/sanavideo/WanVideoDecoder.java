// Portions adapted from Diffusers (Apache-2.0); see META-INF/licenses/sana-video-NOTICE.txt.
package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.nn.ModelMemory;
import io.github.kirstenali.deepj.nn.ModelOps;
import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.persistence.HuggingFaceConfig;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.awt.image.BufferedImage;

/** Wan 2.1 causal video VAE decoder. Each latent frame produces one, then four RGB frames. */
public final class WanVideoDecoder {
    private final HuggingFaceConfig config;
    private final TensorWeights weights;

    public WanVideoDecoder(Path configuration, TensorWeights weights) throws IOException {
        config = new HuggingFaceConfig(configuration);
        config.require("_class_name", "AutoencoderKLWan");
        config.require("is_residual", false);
        if (!config.isNull("patch_size") || !config.array("attn_scales").isEmpty()) {
            throw new IllegalArgumentException("Expected unpatched Wan 2.1 decoder with mid-block attention");
        }
        this.weights = weights;
    }

    public Tensor forward(Tensor latent) {
        List<Tensor> chunks = new ArrayList<>();
        try (var scope = TensorScope.open()) {
            decode(latent, chunk -> chunks.add(retainChunk(chunk)));
            return scope.escape(VideoMemory.finish(ModelOps.cat(2, chunks.toArray(Tensor[]::new))));
        } finally {
            chunks.forEach(Tensor::releaseDeviceBufferRetention);
        }
    }

    /** Converts decoded frames immediately, avoiding a second full floating-point video in memory. */
    public List<BufferedImage> images(Tensor latent) {
        List<BufferedImage> frames = new ArrayList<>();
        try (var ignored = GradMode.noGrad()) {
            decode(latent, chunk -> frames.addAll(VideoFrames.rgb(chunk)));
        }
        return List.copyOf(frames);
    }

    private void decode(Tensor latent, Consumer<Tensor> output) {
        validate(latent);
        try (var scope = TensorScope.open(); var pinned = ModelMemory.retain(latent);
             var cache = new WanFeatureCache(weights)) {
            var blocks = new WanDecoderBlocks(weights, cache);
            Tensor projected = cache.convolution(latent, "post_quant_conv");
            try (var fixed = ModelMemory.retain(projected)) {
                for (int frame = 0; frame < latent.size(2); frame++) {
                    ModelMemory.checkCancelled();
                    chunk(projected.slice(2, frame, frame + 1), cache, blocks, output);
                }
            }
        }
    }

    private static Tensor retainChunk(Tensor chunk) {
        if (GradMode.isEnabled()) chunk.materialize();
        return chunk.retainDeviceBuffer();
    }

    private void validate(Tensor latent) {
        if (latent.rank() != 5 || latent.dtype() != DType.FLOAT32 || latent.size(1) != config.integer("z_dim")) {
            throw new IllegalArgumentException("Expected FLOAT32 channel-first Wan video latents");
        }
    }

    private void chunk(Tensor input, WanFeatureCache cache, WanDecoderBlocks blocks, Consumer<Tensor> output) {
        try (var scope = TensorScope.open()) {
            Tensor hidden = cache.convolution(input, "decoder.conv_in");
            hidden = blocks.residual(hidden, "decoder.mid_block.resnets.0");
            hidden = blocks.attention(hidden, "decoder.mid_block.attentions.0");
            hidden = blocks.residual(hidden, "decoder.mid_block.resnets.1");
            stages(hidden, cache, blocks, 0, output);
        }
    }

    private void stages(Tensor input, WanFeatureCache cache, WanDecoderBlocks blocks,
                        int index, Consumer<Tensor> output) {
        try (var scope = TensorScope.open(); var pinned = ModelMemory.retain(input)) {
            ModelMemory.checkCancelled();
            int count = config.integers("dim_mult").length;
            if (index == count) {
                Tensor hidden = cache.convolution(blocks.activation(input, "decoder.norm_out"), "decoder.conv_out");
                output.accept(VideoMemory.finish(hidden.clamp(-1, 1)));
                return;
            }
            Tensor hidden = VideoMemory.finish(upBlock(input, blocks, index, count));
            try (var fixed = ModelMemory.retain(hidden)) {
                int length = GradMode.isEnabled() ? hidden.size(2) : 1;
                for (int frame = 0; frame < hidden.size(2); frame += length) {
                    stages(hidden.slice(2, frame, frame + length), cache, blocks, index + 1, output);
                }
            }
        }
    }

    private Tensor upBlock(Tensor input, WanDecoderBlocks blocks, int index, int count) {
        String name = "decoder.up_blocks." + index;
        Tensor hidden = input;
        for (int block = 0; block <= config.integer("num_res_blocks"); block++) {
            hidden = blocks.residual(hidden, name + ".resnets." + block);
        }
        if (index == count - 1) return hidden;
        boolean temporal = Boolean.TRUE.equals(config.array("temperal_downsample").get(count - 2 - index));
        return blocks.upsample(hidden, name + ".upsamplers.0", temporal);
    }

    public Tensor denormalize(Tensor latent) {
        float[] mean = values("latents_mean"), deviation = values("latents_std");
        int channels = config.integer("z_dim");
        return latent.multiply(Tensor.from(deviation, 1, channels, 1, 1, 1))
                .add(Tensor.from(mean, 1, channels, 1, 1, 1));
    }

    private float[] values(String name) {
        List<?> list = config.array(name);
        float[] result = new float[list.size()];
        for (int i = 0; i < result.length; i++) { result[i] = ((Number) list.get(i)).floatValue(); }
        return result;
    }
}
