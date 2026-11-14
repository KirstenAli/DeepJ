package io.github.kirstenali.deepj.nn.transformer.qwen3;

import java.util.Map;

final class Qwen3GgufNames {
    private static final Map<String, String> LAYER = Map.ofEntries(
            Map.entry("input_layernorm.gamma", "attn_norm"),
            Map.entry("post_attention_layernorm.gamma", "ffn_norm"),
            Map.entry("self_attn.q_norm.gamma", "attn_q_norm"),
            Map.entry("self_attn.k_norm.gamma", "attn_k_norm"),
            Map.entry("self_attn.q_proj.weight", "attn_q"),
            Map.entry("self_attn.k_proj.weight", "attn_k"),
            Map.entry("self_attn.v_proj.weight", "attn_v"),
            Map.entry("self_attn.o_proj.weight", "attn_output"),
            Map.entry("mlp.gate.weight", "ffn_gate"),
            Map.entry("mlp.up.weight", "ffn_up"),
            Map.entry("mlp.down.weight", "ffn_down"));

    private Qwen3GgufNames() {}

    static String weight(String name) {
        if (name.equals("model.embed_tokens.weight")) return "token_embd.weight";
        if (name.equals("model.norm.gamma")) return "output_norm.weight";
        String[] components = name.split("\\.", 4);
        return "blk." + components[2] + "." + LAYER.get(components[3]) + ".weight";
    }
}
