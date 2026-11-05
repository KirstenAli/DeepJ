package io.github.kirstenali.deepj.nn.transformer.gemma2;

import io.github.kirstenali.deepj.tensor.TensorDescriptor;
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

/** Checks every source name and shape before allocating pretrained parameters. */
final class Gemma2Layout {
    private Gemma2Layout() {}

    static void validate(Gemma2Config config, Map<String, TensorDescriptor> source) throws IOException {
        String prefix = source.containsKey("model.embed_tokens.weight") ? "model." : "";
        Map<String, int[]> expected = inventory(config, prefix);
        if (source.containsKey("lm_head.weight")) {
            expected.put("lm_head.weight", new int[]{config.vocabSize(), config.dModel()});
        }
        if (!source.keySet().equals(expected.keySet())) throw new IOException("Gemma 2 weight inventory mismatch");
        for (var entry : expected.entrySet()) {
            if (!Arrays.equals(entry.getValue(), source.get(entry.getKey()).shape())) {
                throw new IOException("Incorrect Gemma 2 weight shape: " + entry.getKey());
            }
        }
    }

    private static Map<String, int[]> inventory(Gemma2Config config, String prefix) {
        Map<String, int[]> result = new LinkedHashMap<>();
        result.put(prefix + "embed_tokens.weight", new int[]{config.vocabSize(), config.dModel()});
        result.put(prefix + "norm.weight", new int[]{config.dModel()});
        for (int layer = 0; layer < config.nLayers(); layer++) {
            layer(result, prefix + "layers." + layer + ".", config);
        }
        return result;
    }

    private static void layer(Map<String, int[]> shapes, String prefix, Gemma2Config c) {
        for (String name : new String[]{"input_layernorm", "post_attention_layernorm",
                "pre_feedforward_layernorm", "post_feedforward_layernorm"}) {
            shapes.put(prefix + name + ".weight", new int[]{c.dModel()});
        }
        shapes.put(prefix + "self_attn.q_proj.weight", new int[]{c.nHeads() * c.headDim(), c.dModel()});
        shapes.put(prefix + "self_attn.k_proj.weight", new int[]{c.kvHeads() * c.headDim(), c.dModel()});
        shapes.put(prefix + "self_attn.v_proj.weight", new int[]{c.kvHeads() * c.headDim(), c.dModel()});
        shapes.put(prefix + "self_attn.o_proj.weight", new int[]{c.dModel(), c.nHeads() * c.headDim()});
        shapes.put(prefix + "mlp.gate_proj.weight", new int[]{c.dFF(), c.dModel()});
        shapes.put(prefix + "mlp.up_proj.weight", new int[]{c.dFF(), c.dModel()});
        shapes.put(prefix + "mlp.down_proj.weight", new int[]{c.dModel(), c.dFF()});
    }
}
