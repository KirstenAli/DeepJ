package io.github.kirstenali.deepj.models.deepseek;

import io.github.kirstenali.deepj.models.TransformerConfig;

/**
 * Configuration for a DeepSeek-style decoder-only transformer.
 *
 * <p>Extends the shared {@link TransformerConfig} fields with two MLA-specific ranks:
 * <ul>
 *   <li>{@code qRank}  — Q latent dimension (e.g. {@code dModel / 2})</li>
 *   <li>{@code kvRank} — KV latent dimension (e.g. {@code dModel / 4}); controls KV-cache size</li>
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
        double gradClipNorm
) implements TransformerConfig {

    /** Convenience constructor with {@code gradClipNorm = 1.0}. */
    public DeepSeekConfig(int vocabSize, int maxSeqLen, int dModel, int nHeads,
                          int nLayers, int dFF, int qRank, int kvRank) {
        this(vocabSize, maxSeqLen, dModel, nHeads, nLayers, dFF, qRank, kvRank, 1.0);
    }

    public DeepSeekConfig {
        TransformerConfig.validateCommon(vocabSize, maxSeqLen, dModel, nHeads, nLayers, dFF, gradClipNorm);
        if (qRank <= 0)  throw new IllegalArgumentException("qRank must be > 0");
        if (kvRank <= 0) throw new IllegalArgumentException("kvRank must be > 0");
    }
}

