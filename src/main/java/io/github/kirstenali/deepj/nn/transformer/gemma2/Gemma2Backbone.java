package io.github.kirstenali.deepj.nn.transformer.gemma2;

import io.github.kirstenali.deepj.layers.transformer.qwen3.Qwen3Rotary;
import io.github.kirstenali.deepj.nn.ModelMemory;
import io.github.kirstenali.deepj.nn.ModelOps;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.util.function.Function;
import java.util.stream.IntStream;

/** Shared Gemma 2 computation for SANA prompt features and standalone causal language modeling. */
public final class Gemma2Backbone {
    private final Gemma2Config config;
    private final Function<String, Tensor> weights;
    private final Qwen3Rotary rotary;

    public Gemma2Backbone(Gemma2Config config, Function<String, Tensor> weights) {
        this.config = java.util.Objects.requireNonNull(config);
        this.weights = java.util.Objects.requireNonNull(weights);
        rotary = new Qwen3Rotary(config.headDim(), config.ropeTheta());
    }

    public Tensor forward(int[] tokens, int[] valid) { return forward(tokens, valid, null); }

    Tensor forward(int[] tokens, int[] valid, Gemma2Cache cache) {
        requireInput(tokens, valid, cache == null ? 0 : cache.position());
        try (var scope = TensorScope.open()) {
            Tensor input = weights.apply("embed_tokens.weight").indexSelect(0, tokens).unsqueeze(0);
            Tensor scale = Tensor.from(new float[]{(float) Math.sqrt(config.dModel())}, input.dtype(), 1);
            input = input.multiply(scale);
            for (int layer = 0; layer < config.nLayers(); layer++) {
                ModelMemory.checkCancelled();
                input = block(input, valid, layer, cache);
            }
            return scope.escape(normalize(input, "norm"));
        }
    }

    private void requireInput(int[] tokens, int[] valid, int offset) {
        if (tokens.length == 0 || valid.length != tokens.length || tokens.length > config.maxSeqLen() - offset) {
            throw new IllegalArgumentException("Invalid Gemma token sequence or mask");
        }
        for (int token : tokens) {
            if (token < 0 || token >= config.vocabSize()) throw new IllegalArgumentException("Invalid Gemma token ID");
        }
    }

    private Tensor block(Tensor input, int[] valid, int layer, Gemma2Cache cache) {
        String name = "layers." + layer;
        try (var scope = TensorScope.open()) {
            Tensor attended = attention(normalize(input, name + ".input_layernorm"), valid, layer, name, cache);
            Tensor residual = input.add(normalize(attended, name + ".post_attention_layernorm"));
            Tensor hidden = normalize(residual, name + ".pre_feedforward_layernorm");
            Tensor gate = linear(hidden, name + ".mlp.gate_proj").geluActivation();
            hidden = linear(hidden, name + ".mlp.up_proj").multiply(gate);
            Tensor output = residual.add(normalize(linear(hidden, name + ".mlp.down_proj"),
                    name + ".post_feedforward_layernorm"));
            return scope.escape(ModelMemory.finish(output));
        }
    }

    private Tensor attention(Tensor input, int[] valid, int layer, String name, Gemma2Cache cache) {
        String prefix = name + ".self_attn";
        int offset = cache == null ? 0 : cache.position(), length = input.size(1);
        Tensor query = rotary.apply(ModelOps.heads(linear(input, prefix + ".q_proj"), config.nHeads()), length, offset);
        Tensor key = rotary.apply(ModelOps.heads(linear(input, prefix + ".k_proj"), config.kvHeads()), length, offset);
        Tensor value = ModelOps.heads(linear(input, prefix + ".v_proj"), config.kvHeads());
        if (cache != null) cache.append(layer, key, value);
        Tensor result = offset > 0 ? cache.attend(layer, query)
                : prefill(query, key, value, valid, layer);
        return linear(ModelOps.mergeHeads(result), prefix + ".o_proj");
    }

    private Tensor prefill(Tensor query, Tensor key, Tensor value, int[] valid, int layer) {
        int[] repeated = IntStream.range(0, config.nHeads())
                .map(i -> i / (config.nHeads() / config.kvHeads())).toArray();
        return ModelOps.attention(query, key.indexSelect(1, repeated), value.indexSelect(1, repeated),
                mask(valid, layer), (float) (1 / Math.sqrt(config.queryScalar())), config.attentionSoftcap());
    }

    private Tensor normalize(Tensor input, String name) {
        return ModelOps.rms(input, weights.apply(name + ".weight"), null, config.epsilon(), true);
    }

    private Tensor linear(Tensor input, String name) {
        return input.matmul(weights.apply(name + ".weight").transpose(-1, -2));
    }

    private Tensor mask(int[] valid, int layer) {
        int length = valid.length, window = layer % 2 == 0 ? config.slidingWindow() : Integer.MAX_VALUE;
        float[] bias = new float[Math.multiplyExact(length, length)];
        for (int query = 0; query < length; query++) {
            for (int key = 0; key < length; key++) {
                if (key > query || valid[key] == 0 || query - key >= window) bias[query * length + key] = -1e30f;
            }
        }
        return Tensor.from(bias, DType.FLOAT32, 1, 1, length, length);
    }
}
