package io.github.kirstenali.deepj.nn.transformer.gemma2;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.autograd.NormalizationOps;
import io.github.kirstenali.deepj.layers.transformer.qwen3.Qwen3RotaryTables;
import io.github.kirstenali.deepj.nn.ModelMemory;
import io.github.kirstenali.deepj.nn.ModelInference;
import io.github.kirstenali.deepj.nn.ModelOps;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.util.function.Function;
import java.util.Arrays;
import java.util.stream.IntStream;

/** Shared Gemma 2 computation for SANA prompt features and standalone causal language modeling. */
public final class Gemma2Backbone {
    private final Gemma2Config config;
    private final Function<String, Tensor> weights;
    private final Gemma2Rotary rotary;

    public Gemma2Backbone(Gemma2Config config, Function<String, Tensor> weights) {
        this.config = java.util.Objects.requireNonNull(config);
        this.weights = java.util.Objects.requireNonNull(weights);
        rotary = new Gemma2Rotary(config.headDim(), config.ropeTheta());
    }

    public Tensor forward(int[] tokens, int[] valid) { return forward(tokens, valid, null); }

    /** Encodes equal-length padded sequences together, keeping each row's causal mask independent. */
    public Tensor forward(int[][] tokens, int[][] valid) {
        if (tokens.length == 0 || valid.length != tokens.length) {
            throw new IllegalArgumentException("Gemma batches require matching token and mask rows");
        }
        int length = tokens[0].length;
        for (int row = 0; row < tokens.length; row++) {
            requireInput(tokens[row], valid[row], 0);
            if (tokens[row].length != length) throw new IllegalArgumentException("Gemma rows must have equal length");
        }
        return evaluate(flatten(tokens), flatten(valid), null, tokens.length);
    }

    private static int[] flatten(int[][] rows) {
        return Arrays.stream(rows).flatMapToInt(Arrays::stream).toArray();
    }

    Tensor forward(int[] tokens, int[] valid, Gemma2Cache cache) {
        requireInput(tokens, valid, cache == null ? 0 : cache.position());
        return evaluate(tokens, valid, cache, 1);
    }

    private Tensor evaluate(int[] tokens, int[] valid, Gemma2Cache cache, int batch) {
        try (var scope = TensorScope.open(); var tables = Qwen3RotaryTables.open()) {
            Tensor input = weights.apply("embed_tokens.weight").indexSelect(0, tokens)
                    .reshape(batch, tokens.length / batch, config.dModel());
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
        if (cache == null) return ModelInference.evaluate(() -> blockValues(input, valid, layer, null));
        return blockValues(input, valid, layer, cache);
    }

    private Tensor blockValues(Tensor input, int[] valid, int layer, Gemma2Cache cache) {
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
        return Gemma2Attention.apply(query, key.indexSelect(1, repeated), value.indexSelect(1, repeated),
                mask(valid, layer, query.size(0)), (float) (1 / Math.sqrt(config.queryScalar())),
                config.attentionSoftcap());
    }

    private Tensor normalize(Tensor input, String name) {
        Tensor gain = weights.apply(name + ".weight");
        if (!GradMode.isEnabled()) return ModelOps.rms(input, gain, null, config.epsilon(), true);
        Tensor matrix = input.to(DType.FLOAT32).reshape(-1, input.size(-1));
        Tensor scale = gain.to(DType.FLOAT32).addScalar(1).reshape(1, -1);
        return NormalizationOps.rmsNorm(matrix, scale, config.epsilon()).reshape(input.shape()).to(input.dtype());
    }

    private Tensor linear(Tensor input, String name) {
        Tensor weight = weights.apply(name + ".weight").transpose(-1, -2);
        if (!GradMode.isEnabled()) return input.matmul(weight);
        return input.reshape(-1, input.size(-1)).matmul(weight)
                .reshape(input.size(0), input.size(1), weight.size(-1));
    }

    private Tensor mask(int[] valid, int layer, int batch) {
        int length = valid.length / batch, window = layer % 2 == 0 ? config.slidingWindow() : Integer.MAX_VALUE;
        float[] bias = new float[Math.multiplyExact(batch, Math.multiplyExact(length, length))];
        for (int row = 0; row < batch; row++) { maskRow(bias, valid, row, length, window); }
        return Tensor.from(bias, DType.FLOAT32, batch, 1, length, length);
    }

    private static void maskRow(float[] bias, int[] valid, int batch, int length, int window) {
        for (int query = 0; query < length; query++) {
            for (int key = 0; key < length; key++) {
                if (key > query || valid[batch * length + key] == 0 || query - key >= window) {
                    bias[(batch * length + query) * length + key] = -1e30f;
                }
            }
        }
    }
}
