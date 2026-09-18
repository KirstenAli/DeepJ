package io.github.kirstenali.deepj.models.deepseek;

import io.github.kirstenali.deepj.models.TransformerConfig;

/**
 * Configuration for a DeepSeek-style decoder-only transformer.
 *
 * <p>Extends the shared {@link TransformerConfig} fields with two MLA-specific ranks:
 * <ul>
 *   <li>{@code qRank}  — Q latent dimension (e.g. {@code dModel / 2})</li>
 *   <li>{@code kvRank} — KV latent dimension (e.g. {@code dModel / 4})</li>
 * </ul>
 */
public record DeepSeekConfig(
        int vocabSize,
        int maxSeqLen,
        int dModel,
        int nHeads,
        int nLayers,
        int dFF,
        int qRank,
        int kvRank,
        float initScale,
        float gradClipNorm
) implements TransformerConfig {

    /** Convenience constructor with stable initialization and gradient-clipping defaults. */
    public DeepSeekConfig(int vocabSize, int maxSeqLen, int dModel, int nHeads,
                          int nLayers, int dFF, int qRank, int kvRank) {
        this(vocabSize, maxSeqLen, dModel, nHeads, nLayers, dFF, qRank, kvRank, 0.2f, 1.0f);
    }

    /** Convenience constructor retaining the historical gradient-clip argument. */
    public DeepSeekConfig(int vocabSize, int maxSeqLen, int dModel, int nHeads,
                          int nLayers, int dFF, int qRank, int kvRank, float gradClipNorm) {
        this(vocabSize, maxSeqLen, dModel, nHeads, nLayers, dFF, qRank, kvRank, 0.2f, gradClipNorm);
    }

    public DeepSeekConfig {
        TransformerConfig.validateCommon(vocabSize, maxSeqLen, dModel, nHeads, nLayers, dFF, gradClipNorm);
        TransformerConfig.validateRotaryHeadDimension(dModel, nHeads);
        if (qRank <= 0)  throw new IllegalArgumentException("qRank must be > 0");
        if (kvRank <= 0) throw new IllegalArgumentException("kvRank must be > 0");
        if (!Float.isFinite(initScale) || initScale <= 0.0f) {
            throw new IllegalArgumentException("initScale must be finite and > 0");
        }
    }
}
