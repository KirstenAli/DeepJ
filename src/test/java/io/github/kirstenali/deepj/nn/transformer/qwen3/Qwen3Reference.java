package io.github.kirstenali.deepj.nn.transformer.qwen3;

import java.util.HashMap;
import java.util.Map;

final class Qwen3Reference {
    private final Qwen3Config config;
    private final Map<String, double[]> weights = new HashMap<>();

    Qwen3Reference(Qwen3 model) {
        config = model.config();
        model.parameters().forEach(parameter ->
                weights.put(parameter.name(), ReferenceMath.doubles(parameter.value.toFloatArray())));
    }

    float[] logits(int[] tokens, int window) {
        double[] embeddings = weight("model.embed_tokens.weight");
        double[][] hidden = new double[tokens.length][config.dModel()];
        for (int row = 0; row < hidden.length; row++) {
            System.arraycopy(embeddings, tokens[row] * config.dModel(), hidden[row], 0, config.dModel());
        }
        for (int layer = 0; layer < config.nLayers(); layer++) { hidden = block(layer, hidden, window); }
        return ReferenceMath.floats(project(ReferenceMath.norm(hidden, weight("model.norm.gamma")), embeddings));
    }

    private double[][] block(int layer, double[][] input, int window) {
        String prefix = "model.layers." + layer;
        double[][] normalized = ReferenceMath.norm(input, weight(prefix + ".input_layernorm.gamma"));
        double[][] residual = ReferenceMath.add(input, attention(prefix + ".self_attn", normalized, window));
        double[][] norm = ReferenceMath.norm(residual, weight(prefix + ".post_attention_layernorm.gamma"));
        double[][] gate = ReferenceMath.linear(norm, weight(prefix + ".mlp.gate.weight"), config.dFF());
        double[][] up = ReferenceMath.linear(norm, weight(prefix + ".mlp.up.weight"), config.dFF());
        return ReferenceMath.add(residual, ReferenceMath.linear(ReferenceMath.gated(gate, up),
                weight(prefix + ".mlp.down.weight"), config.dModel()));
    }

    private double[][] attention(String prefix, double[][] input, int window) {
        double[][] query = ReferenceMath.linear(input, weight(prefix + ".q_proj.weight"),
                config.nHeads() * config.headDim());
        double[][] key = ReferenceMath.linear(input, weight(prefix + ".k_proj.weight"),
                config.kvHeads() * config.headDim());
        double[][] value = ReferenceMath.linear(input, weight(prefix + ".v_proj.weight"),
                config.kvHeads() * config.headDim());
        query = ReferenceAttention.normalized(query, weight(prefix + ".q_norm.gamma"), config.ropeTheta());
        key = ReferenceAttention.normalized(key, weight(prefix + ".k_norm.gamma"), config.ropeTheta());
        return ReferenceMath.linear(ReferenceAttention.apply(config, query, key, value, window),
                weight(prefix + ".o_proj.weight"), config.dModel());
    }

    private double[][] project(double[][] input, double[] embedding) {
        double[][] result = new double[input.length][config.vocabSize()];
        for (int row = 0; row < input.length; row++) { projectRow(input[row], embedding, result[row]); }
        return result;
    }

    private void projectRow(double[] input, double[] embedding, double[] output) {
        for (int token = 0; token < output.length; token++) {
            for (int dimension = 0; dimension < input.length; dimension++) {
                output[token] += input[dimension] * embedding[token * input.length + dimension];
            }
        }
    }

    private double[] weight(String name) {
        double[] result = weights.get(name);
        if (result == null) throw new IllegalArgumentException("Missing reference weight " + name);
        return result;
    }
}
