package io.github.kirstenali.deepj.models.llama;

import io.github.kirstenali.deepj.models.TransformerConfig;

public record LlamaConfig(
        int vocabSize,
        int maxSeqLen,
        int dModel,
        int nHeads,
        int nLayers,
        int dFF,
        float gradClipNorm
) implements TransformerConfig {

    /** Convenience constructor with {@code gradClipNorm = 1.0}. */
    public LlamaConfig(int vocabSize, int maxSeqLen, int dModel, int nHeads, int nLayers, int dFF) {
        this(vocabSize, maxSeqLen, dModel, nHeads, nLayers, dFF, 1.0f);
    }

    public LlamaConfig {
        TransformerConfig.validateCommon(vocabSize, maxSeqLen, dModel, nHeads, nLayers, dFF, gradClipNorm);
    }

    /**
     * Returns the standard Llama dFF for the given dModel:
     * {@code round(8/3 * dModel)} rounded up to the nearest multiple of 64.
     */
    public static int defaultDFF(int dModel) {
        int raw = (int) Math.round(8.0 / 3 * dModel);
        return ((raw + 63) / 64) * 64;
    }
}
