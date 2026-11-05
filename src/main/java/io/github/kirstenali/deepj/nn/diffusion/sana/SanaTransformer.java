package io.github.kirstenali.deepj.nn.diffusion.sana;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.io.IOException;
import java.nio.file.Path;

/** SANA-Sprint's conditioned linear-attention diffusion transformer. */
public final class SanaTransformer {
    private final SanaConfig config;
    private final SanaWeights weights;
    private final int width;

    public SanaTransformer(Path config, SanaWeights weights) throws IOException {
        this.config = new SanaConfig(config);
        this.config.require("_class_name", "SanaTransformer2DModel");
        this.config.require("patch_size", 1L);
        if (!this.config.flag("guidance_embeds") || !this.config.isNull("interpolation_scale")) {
            throw new IllegalArgumentException("Expected SANA-Sprint guidance and position configuration");
        }
        this.weights = weights;
        width = this.config.integer("num_attention_heads") * this.config.integer("attention_head_dim");
    }

    public Tensor forward(Tensor latent, Tensor text, int[] valid, float time, float guidance) {
        if (latent.rank() != 4 || latent.size(0) != 1 || latent.size(1) != config.integer("in_channels")
                || text.rank() != 3 || text.size(0) != 1 || text.size(1) != valid.length) {
            throw new IllegalArgumentException("Invalid SANA latent, prompt features, or mask");
        }
        try (var scope = TensorScope.open(); var inputs = SanaMemory.retain(latent, text)) {
            Tensor hidden = SanaOps.imageTokens(SanaConvolution.apply(latent, weights, "patch_embed.proj"));
            Tensor conditioning = conditioning(time, guidance).to(hidden.dtype());
            Tensor modulation = SanaOps.linear(SanaOps.silu(conditioning), weights, "time_embed.linear");
            Tensor caption = caption(text), mask = textMask(valid);
            try (var pinned = SanaMemory.retain(conditioning, modulation, caption, mask)) {
                for (int layer = 0; layer < config.integer("num_layers"); layer++) {
                    SanaMemory.checkCancelled();
                    hidden = block(hidden, caption, mask, modulation, layer, latent.size(2), latent.size(3));
                }
                return scope.escape(output(hidden, conditioning, latent.size(2), latent.size(3)));
            }
        }
    }

    private Tensor caption(Tensor text) {
        Tensor hidden = SanaOps.linear(text, weights, "caption_projection.linear_1").geluActivation();
        hidden = SanaOps.linear(hidden, weights, "caption_projection.linear_2");
        return SanaOps.affineRms(hidden, weights, "caption_norm", 1e-5f, false);
    }

    private Tensor conditioning(float time, float guidance) {
        Tensor first = timeEmbedding(time, "time_embed.timestep_embedder");
        return first.add(timeEmbedding(guidance * config.decimal("guidance_embeds_scale"),
                "time_embed.guidance_embedder"));
    }

    private Tensor timeEmbedding(float value, String name) {
        float[] features = new float[256];
        for (int index = 0; index < 128; index++) {
            double angle = value * Math.exp(-Math.log(10000) * index / 128);
            features[index] = (float) Math.cos(angle);
            features[index + 128] = (float) Math.sin(angle);
        }
        Tensor hidden = SanaOps.linear(Tensor.from(features, 1, 256), weights, name + ".linear_1");
        return SanaOps.linear(SanaOps.silu(hidden), weights, name + ".linear_2");
    }

    private Tensor block(Tensor input, Tensor text, Tensor mask, Tensor time, int layer, int height, int imageWidth) {
        String name = "transformer_blocks." + layer;
        try (var scope = TensorScope.open()) {
            Tensor parts = weights.tensor(name + ".scale_shift_table").unsqueeze(0).add(time.reshape(1, 6, width));
            Tensor hidden = modulate(input, parts, 0, 1);
            hidden = input.add(selfAttention(hidden, name + ".attn1").multiply(parts.slice(1, 2, 3)));
            hidden = hidden.add(crossAttention(hidden, text, mask, name + ".attn2"));
            Tensor feed = SanaOps.image(modulate(hidden, parts, 3, 4), height, imageWidth);
            feed = SanaConvolution.gated(feed, weights, name + ".ff", false);
            Tensor output = hidden.add(SanaOps.imageTokens(feed).multiply(parts.slice(1, 5, 6)));
            return scope.escape(SanaMemory.finish(output));
        }
    }

    private Tensor modulate(Tensor input, Tensor parts, int shift, int scale) {
        return SanaOps.norm(input, config.decimal("norm_eps"))
                .multiply(parts.slice(1, scale, scale + 1).addScalar(1)).add(parts.slice(1, shift, shift + 1));
    }

    private Tensor selfAttention(Tensor input, String name) {
        int heads = config.integer("num_attention_heads");
        Tensor query = normalizedProjection(input, name, "q"), key = normalizedProjection(input, name, "k");
        Tensor value = SanaOps.linear(input, weights, name + ".to_v");
        Tensor result = SanaOps.linearAttention(SanaOps.heads(query, heads), SanaOps.heads(key, heads),
                SanaOps.heads(value, heads), 1e-15f);
        return SanaOps.linear(SanaOps.mergeHeads(result), weights, name + ".to_out.0");
    }

    private Tensor crossAttention(Tensor input, Tensor text, Tensor mask, String name) {
        int heads = config.integer("num_cross_attention_heads");
        Tensor query = normalizedProjection(input, name, "q"), key = normalizedProjection(text, name, "k");
        Tensor value = SanaOps.linear(text, weights, name + ".to_v");
        Tensor result = SanaOps.attention(SanaOps.heads(query, heads), SanaOps.heads(key, heads),
                SanaOps.heads(value, heads), mask,
                (float) (1 / Math.sqrt(config.integer("cross_attention_head_dim"))), 0);
        return SanaOps.linear(SanaOps.mergeHeads(result), weights, name + ".to_out.0");
    }

    private Tensor normalizedProjection(Tensor input, String name, String kind) {
        Tensor result = SanaOps.linear(input, weights, name + ".to_" + kind);
        return SanaOps.affineRms(result, weights, name + ".norm_" + kind, 1e-5f, false);
    }

    private Tensor output(Tensor hidden, Tensor conditioning, int height, int imageWidth) {
        Tensor parts = weights.tensor("scale_shift_table").unsqueeze(0).add(conditioning.unsqueeze(1));
        Tensor result = SanaOps.norm(hidden, 1e-6f).multiply(parts.slice(1, 1, 2).addScalar(1))
                .add(parts.slice(1, 0, 1));
        return SanaOps.image(SanaOps.linear(result, weights, "proj_out"), height, imageWidth);
    }

    private static Tensor textMask(int[] valid) {
        float[] values = new float[valid.length];
        for (int index = 0; index < valid.length; index++) {
            values[index] = valid[index] == 0 ? -10000 : 0;
        }
        return Tensor.from(values, 1, 1, 1, valid.length);
    }
}
