// Portions adapted from Diffusers (Apache-2.0); see META-INF/licenses/sana-video-NOTICE.txt.
package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.nn.ModelMemory;
import io.github.kirstenali.deepj.nn.ModelOps;
import io.github.kirstenali.deepj.persistence.HuggingFaceConfig;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.io.IOException;
import java.nio.file.Path;

/** SANA-Video's conditioned 3D Transformer, using DeepJ tensors for all network arithmetic. */
public final class SanaVideoTransformer implements AutoCloseable {
    private final HuggingFaceConfig config;
    private final TensorWeights weights;
    private final SanaVideoConditioning conditioning;
    private final SanaVideoBlocks blocks;
    private final boolean owned;

    public SanaVideoTransformer(Path configuration, TensorWeights weights) throws IOException {
        this(configuration, weights, null);
    }

    private SanaVideoTransformer(Path configuration, TensorWeights weights, VideoWeightCache source)
            throws IOException {
        config = new HuggingFaceConfig(configuration);
        config.require("_class_name", "SanaVideoTransformer3DModel");
        config.require("qk_norm", "rms_norm_across_heads");
        config.require("guidance_embeds", false);
        config.require("norm_elementwise_affine", false);
        if (!config.isNull("interpolation_scale")) throw new IllegalArgumentException("Unsupported position scale");
        this.weights = weights;
        conditioning = new SanaVideoConditioning(weights);
        blocks = new SanaVideoBlocks(config, weights, source);
        owned = source != null;
        if (config.integer("num_cross_attention_heads") != config.integer("num_attention_heads")) {
            throw new IllegalArgumentException("Expected matching video self/cross attention heads");
        }
    }

    /** Streams inference blocks; the supplied-weights constructor supports differentiable execution. */
    public static SanaVideoTransformer streamed(Path directory) throws IOException {
        return load(directory, 0);
    }

    /** Loads FP32 working weights with a separate checkpoint-cache budget. Inference only. */
    public static SanaVideoTransformer load(Path directory, long cacheBytes) throws IOException {
        return load(directory, cacheBytes, DType.FLOAT32);
    }

    /** Streams or caches weights in the selected working dtype. Normalization statistics remain FLOAT32. */
    public static SanaVideoTransformer load(Path directory, long cacheBytes, DType dtype) throws IOException {
        var source = new VideoWeightCache(directory, cacheBytes, dtype);
        TensorWeights base = null;
        try {
            base = source.select(name -> !name.startsWith("transformer_blocks."));
            return new SanaVideoTransformer(directory.resolve("config.json"), base, source);
        } catch (IOException | RuntimeException | Error failure) {
            if (base != null) base.close();
            source.close();
            throw failure;
        }
    }

    @Override public void close() {
        if (owned) {
            blocks.close();
            weights.close();
        }
    }

    public Tensor forward(Tensor latent, Tensor text, int[] valid, float timestep) {
        validate(latent, text, valid, timestep);
        int[] patch = config.integers("patch_size");
        int t = latent.size(2) / patch[0], h = latent.size(3) / patch[1], w = latent.size(4) / patch[2];
        try (var scope = TensorScope.open(); var pinned = ModelMemory.retain(latent, text)) {
            var rotary = new VideoRotary(t, h, w, config.integer("attention_head_dim"));
            Tensor hidden = VideoConvolution.patch(latent, weights.tensor("patch_embedding.weight"),
                    weights.tensor("patch_embedding.bias"));
            Tensor embedding = conditioning.time(timestep);
            Tensor modulation = conditioning.modulation(embedding);
            Tensor caption = conditioning.caption(text), mask = SanaVideoConditioning.mask(valid);
            try (var positions = ModelMemory.retain(rotary.buffers());
                 var fixed = ModelMemory.retain(embedding, modulation, caption, mask)) {
                for (int index = 0; index < config.integer("num_layers"); index++) {
                    hidden = blocks.forward(hidden, caption, mask, modulation, rotary, index, new int[]{t, h, w});
                }
                return scope.escape(output(hidden, embedding, patch, t, h, w));
            }
        }
    }

    private Tensor output(Tensor input, Tensor embedding, int[] patch, int t, int h, int w) {
        Tensor affine = weights.tensor("scale_shift_table").unsqueeze(0)
                .add(embedding.unsqueeze(1));
        Tensor normalized = ModelOps.norm(input, 1e-6f).multiply(affine.slice(1, 1, 2).addScalar(1))
                .add(affine.slice(1, 0, 1));
        return ModelOps.linear(normalized, weights, "proj_out")
                .reshape(1, t, h, w, patch[0], patch[1], patch[2], config.integer("out_channels"))
                .permute(0, 7, 1, 4, 2, 5, 3, 6)
                .reshape(1, config.integer("out_channels"), t * patch[0], h * patch[1], w * patch[2]);
    }

    private void validate(Tensor latent, Tensor text, int[] mask, float time) {
        if (latent.dtype() != weights.tensor("patch_embedding.weight").dtype() || text.dtype() != latent.dtype()) {
            throw new IllegalArgumentException("Video Transformer inputs must match its weight dtype");
        }
        if (latent.rank() != 5 || latent.size(0) != 1 || latent.size(1) != config.integer("in_channels")) {
            throw new IllegalArgumentException("Expected one channel-first video latent");
        }
        validateCaption(text, mask, time);
        validateGrid(latent);
    }

    private void validateCaption(Tensor text, int[] mask, float time) {
        if (text.rank() != 3 || text.size(0) != 1 || text.size(1) != mask.length
                || text.size(2) != config.integer("caption_channels") || !Float.isFinite(time)) {
            throw new IllegalArgumentException("Invalid video caption, mask, or timestep");
        }
    }

    private void validateGrid(Tensor latent) {
        int[] patch = config.integers("patch_size");
        for (int axis = 0; axis < 3; axis++) {
            if (latent.size(axis + 2) % patch[axis] != 0
                    || latent.size(axis + 2) / patch[axis] > config.integer("rope_max_seq_len")) {
                throw new IllegalArgumentException("Latent dimensions exceed or do not fit the video patch grid");
            }
        }
    }
}
