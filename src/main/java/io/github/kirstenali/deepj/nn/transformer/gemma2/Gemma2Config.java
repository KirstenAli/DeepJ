package io.github.kirstenali.deepj.nn.transformer.gemma2;

import io.github.kirstenali.deepj.nn.transformer.TransformerConfig;
import io.github.kirstenali.deepj.persistence.HuggingFaceConfig;
import java.io.IOException;
import java.nio.file.Path;

/** Gemma 2 geometry, including its independent head width and alternating local/global attention. */
public record Gemma2Config(int vocabSize, int dModel, int dFF, int nHeads, int kvHeads, int nLayers,
                            int headDim, int maxSeqLen, int slidingWindow, float epsilon,
                            float queryScalar, float attentionSoftcap, float logitSoftcap, double ropeTheta)
        implements TransformerConfig {
    public Gemma2Config {
        positive(vocabSize, dModel, dFF, nHeads, kvHeads, nLayers, headDim, maxSeqLen, slidingWindow);
        positive(epsilon, queryScalar, attentionSoftcap, logitSoftcap, ropeTheta);
        if (nHeads % kvHeads != 0 || headDim % 2 != 0) {
            throw new IllegalArgumentException("Invalid Gemma 2 head geometry");
        }
    }

    private static void positive(double... values) {
        for (double value : values) {
            if (!(value > 0) || !Double.isFinite(value)) {
                throw new IllegalArgumentException("Gemma 2 dimensions and scales must be positive and finite");
            }
        }
    }

    public static Gemma2Config read(Path path) throws IOException {
        var config = new HuggingFaceConfig(path);
        config.require("model_type", "gemma2");
        config.require("hidden_activation", "gelu_pytorch_tanh");
        config.require("attention_bias", false);
        return new Gemma2Config(config.integer("vocab_size"), config.integer("hidden_size"),
                config.integer("intermediate_size"), config.integer("num_attention_heads"),
                config.integer("num_key_value_heads"), config.integer("num_hidden_layers"),
                config.integer("head_dim"), config.integer("max_position_embeddings"),
                config.integer("sliding_window"), config.decimal("rms_norm_eps"),
                config.decimal("query_pre_attn_scalar"), config.decimal("attn_logit_softcapping"),
                config.decimal("final_logit_softcapping"), config.decimal("rope_theta"));
    }

    @Override public float initScale() { return 1; }
    @Override public float gradClipNorm() { return 1; }
}
