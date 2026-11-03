package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Files;
import java.util.LinkedHashMap;
import java.util.Map;

public final class Qwen3ConfigIO {
    private Qwen3ConfigIO() {}

    public static void write(Path path, Qwen3Config config) throws IOException {
        Map<String, Object> values = new LinkedHashMap<>();
        values.putAll(Map.of("model_type", "qwen3", "architectures", java.util.List.of("Qwen3ForCausalLM"),
                "tie_word_embeddings", true, "hidden_act", "silu", "attention_bias", false));
        values.putAll(Map.of("vocab_size", config.vocabSize(), "max_position_embeddings", config.maxSeqLen(),
                "hidden_size", config.dModel(), "num_attention_heads", config.nHeads(),
                "num_key_value_heads", config.kvHeads(), "num_hidden_layers", config.nLayers(),
                "intermediate_size", config.dFF(), "head_dim", config.headDim()));
        values.putAll(Map.of("rope_theta", config.ropeTheta(), "rms_norm_eps", config.rmsNormEpsilon(),
                "initializer_range", config.initializerRange(), "gradient_clip_norm", config.gradClipNorm(),
                "attention_dropout", 0, "use_sliding_window", false));
        Files.writeString(path, Qwen3Json.stringify(values));
    }

    public static Qwen3Config read(Path path) throws IOException {
        try { return parse(Qwen3Json.read(path)); }
        catch (IllegalArgumentException | ArithmeticException failure) {
            throw new IOException("Unsupported or invalid Qwen3 configuration", failure);
        }
    }

    public static Qwen3Config parse(Map<String, Object> values) {
        validateOptions(values);
        return new Qwen3Config(integer(values, "vocab_size"), integer(values, "max_position_embeddings"),
                integer(values, "hidden_size"), integer(values, "num_attention_heads"),
                integer(values, "num_key_value_heads"), integer(values, "num_hidden_layers"),
                integer(values, "intermediate_size"), integer(values, "head_dim"),
                number(values, "rope_theta", 1_000_000), (float) number(values, "initializer_range", 0.02),
                (float) number(values, "gradient_clip_norm", 1));
    }

    private static void validateOptions(Map<String, Object> values) {
        if (!"qwen3".equals(values.get("model_type")) || !"silu".equals(values.get("hidden_act"))
                || !Boolean.TRUE.equals(values.get("tie_word_embeddings"))) {
            throw new IllegalArgumentException("Expected a tied-embedding Qwen3 SiLU decoder");
        }
        validateAttention(values);
        if ((float) number(values, "rms_norm_eps", 1e-6) != 1e-6f) {
            throw new IllegalArgumentException("Expected the released Qwen3 normalization epsilon");
        }
    }

    private static void validateAttention(Map<String, Object> values) {
        if (Boolean.TRUE.equals(values.get("attention_bias")) || Boolean.TRUE.equals(values.get("use_sliding_window"))
                || values.get("rope_scaling") != null || number(values, "attention_dropout", 0) != 0) {
            throw new IllegalArgumentException("Only bias-free full attention and default RoPE are supported");
        }
    }

    private static int integer(Map<String, Object> values, String name) { return Qwen3Json.integer(values, name); }
    private static double number(Map<String, Object> values, String name, double fallback) {
        return values.containsKey(name) ? Qwen3Json.number(values.get(name)) : fallback;
    }
}
