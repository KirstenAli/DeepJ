package io.github.kirstenali.deepj.nn.diffusion.sana;

import io.github.kirstenali.deepj.layers.transformer.qwen3.Qwen3Rotary;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.io.IOException;
import java.nio.file.Path;
import java.util.stream.IntStream;

/** Gemma 2 prompt features, including its four normalization sites and attention softcap. */
public final class Gemma2Encoder {
    private final SanaWeights weights;
    private final SanaConfig config;
    private final Qwen3Rotary rotary;

    public Gemma2Encoder(Path config, SanaWeights weights) throws IOException {
        this.config = new SanaConfig(config);
        this.config.require("model_type", "gemma2");
        this.weights = weights;
        rotary = new Qwen3Rotary(this.config.integer("head_dim"), this.config.decimal("rope_theta"));
    }

    public Tensor forward(int[] tokens, int[] valid) {
        if (tokens.length == 0 || valid.length != tokens.length
                || tokens.length > config.integer("max_position_embeddings")) {
            throw new IllegalArgumentException("Invalid Gemma token sequence or mask");
        }
        try (var scope = TensorScope.open()) {
            Tensor input = weights.tensor("embed_tokens.weight").indexSelect(0, tokens).unsqueeze(0);
            input = input.multiplyScalar((float) Math.sqrt(config.integer("hidden_size")));
            for (int layer = 0; layer < config.integer("num_hidden_layers"); layer++) {
                input = block(input, valid, layer);
            }
            return scope.escape(normalize(input, "norm"));
        }
    }

    private Tensor block(Tensor input, int[] valid, int layer) {
        String name = "layers." + layer;
        try (var scope = TensorScope.open()) {
            Tensor attention = attention(normalize(input, name + ".input_layernorm"), valid, layer, name);
            Tensor residual = input.add(normalize(attention, name + ".post_attention_layernorm"));
            Tensor hidden = normalize(residual, name + ".pre_feedforward_layernorm");
            Tensor gate = SanaOps.linear(hidden, weights, name + ".mlp.gate_proj").geluActivation();
            hidden = gate.multiply(SanaOps.linear(hidden, weights, name + ".mlp.up_proj"));
            hidden = SanaOps.linear(hidden, weights, name + ".mlp.down_proj");
            Tensor output = residual.add(normalize(hidden, name + ".post_feedforward_layernorm"));
            return scope.escape(SanaMemory.finish(output));
        }
    }

    private Tensor attention(Tensor input, int[] valid, int layer, String name) {
        String prefix = name + ".self_attn";
        int heads = config.integer("num_attention_heads"), keys = config.integer("num_key_value_heads");
        Tensor query = rotary.apply(SanaOps.heads(SanaOps.linear(input, weights, prefix + ".q_proj"), heads),
                input.size(1), 0);
        Tensor key = SanaOps.heads(SanaOps.linear(input, weights, prefix + ".k_proj"), keys);
        key = rotary.apply(key, input.size(1), 0);
        Tensor value = SanaOps.heads(SanaOps.linear(input, weights, prefix + ".v_proj"), keys);
        int[] repeated = IntStream.range(0, heads).map(i -> i / (heads / keys)).toArray();
        Tensor result = SanaOps.attention(query, key.indexSelect(1, repeated), value.indexSelect(1, repeated),
                mask(valid, layer), (float) (1 / Math.sqrt(config.decimal("query_pre_attn_scalar"))),
                config.decimal("attn_logit_softcapping"));
        return SanaOps.linear(SanaOps.mergeHeads(result), weights, prefix + ".o_proj");
    }

    private Tensor normalize(Tensor input, String name) {
        return SanaOps.affineRms(input, weights, name, config.decimal("rms_norm_eps"), true);
    }

    private Tensor mask(int[] valid, int layer) {
        int length = valid.length;
        int window = layer % 2 == 0 ? config.integer("sliding_window") : Integer.MAX_VALUE;
        float[] bias = new float[Math.multiplyExact(length, length)];
        for (int query = 0; query < length; query++) {
            for (int key = 0; key < length; key++) {
                if (key > query || valid[key] == 0 || query - key >= window) {
                    bias[query * length + key] = -1e30f;
                }
            }
        }
        return Tensor.from(bias, DType.FLOAT32, 1, 1, length, length);
    }
}
