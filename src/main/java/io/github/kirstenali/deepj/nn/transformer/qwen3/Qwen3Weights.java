package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.List;

final class Qwen3Weights {
    private Qwen3Weights() {}

    record Weight(String name, String source, int rows, int columns, boolean transpose, boolean norm) {
        TensorDescriptor descriptor(DType dtype) { return new TensorDescriptor(dtype, new int[]{rows, columns}); }
        int[] sourceShape() {
            if (norm) return new int[]{columns};
            return transpose ? new int[]{columns, rows} : new int[]{rows, columns};
        }
    }

    static List<Weight> inventory(Qwen3Config config) {
        List<Weight> result = new ArrayList<>();
        result.add(new Weight("model.embed_tokens.weight", "model.embed_tokens.weight",
                config.vocabSize(), config.dModel(), false, false));
        for (int index = 0; index < config.nLayers(); index++) {
            block(result, "model.layers." + index, config);
        }
        norm(result, "model.norm", config.dModel());
        return List.copyOf(result);
    }

    private static void block(List<Weight> result, String prefix, Qwen3Config config) {
        norm(result, prefix + ".input_layernorm", config.dModel());
        norm(result, prefix + ".post_attention_layernorm", config.dModel());
        norm(result, prefix + ".self_attn.q_norm", config.headDim());
        norm(result, prefix + ".self_attn.k_norm", config.headDim());
        projection(result, prefix + ".self_attn.q_proj", config.dModel(), config.nHeads() * config.headDim());
        projection(result, prefix + ".self_attn.k_proj", config.dModel(), config.kvHeads() * config.headDim());
        projection(result, prefix + ".self_attn.v_proj", config.dModel(), config.kvHeads() * config.headDim());
        projection(result, prefix + ".self_attn.o_proj", config.nHeads() * config.headDim(), config.dModel());
        mlp(result, prefix, "gate", config.dModel(), config.dFF());
        mlp(result, prefix, "up", config.dModel(), config.dFF());
        mlp(result, prefix, "down", config.dFF(), config.dModel());
    }

    private static void norm(List<Weight> result, String prefix, int width) {
        result.add(new Weight(prefix + ".gamma", prefix + ".weight", 1, width, false, true));
    }

    private static void projection(List<Weight> result, String prefix, int rows, int columns) {
        result.add(new Weight(prefix + ".weight", prefix + ".weight", rows, columns, true, false));
    }

    private static void mlp(List<Weight> result, String prefix, String part, int rows, int columns) {
        result.add(new Weight(prefix + ".mlp." + part + ".weight", prefix + ".mlp." + part + "_proj.weight",
                rows, columns, true, false));
    }
}
