package io.github.kirstenali.deepj.models.llama;

import io.github.kirstenali.deepj.models.TransformerConfig;

public record LlamaConfig(
        int vocabSize,
        int maxSeqLen,
        int dModel,
        int nHeads,
        int nLayers,
        int dFF,
        float initScale,
        float gradClipNorm
) implements TransformerConfig {

    /** Convenience constructor with stable initialization and gradient-clipping defaults. */
    public LlamaConfig(int vocabSize, int maxSeqLen, int dModel, int nHeads, int nLayers, int dFF) {
        this(vocabSize, maxSeqLen, dModel, nHeads, nLayers, dFF, 0.2f, 1.0f);
    }

    /** Convenience constructor retaining the historical gradient-clip argument. */
    public LlamaConfig(int vocabSize, int maxSeqLen, int dModel, int nHeads, int nLayers,
                       int dFF, float gradClipNorm) {
        this(vocabSize, maxSeqLen, dModel, nHeads, nLayers, dFF, 0.2f, gradClipNorm);
    }

    public LlamaConfig {
        TransformerConfig.validateCommon(vocabSize, maxSeqLen, dModel, nHeads, nLayers, dFF, gradClipNorm);
        TransformerConfig.validateRotaryHeadDimension(dModel, nHeads);
        if (!Float.isFinite(initScale) || initScale <= 0.0f) {
            throw new IllegalArgumentException("initScale must be finite and > 0");
        }
    }

    /**
     * Returns a common Llama-style dFF for the given dModel:
     * {@code round(8/3 * dModel)} rounded up to the nearest multiple of 64.
     */
    public static int defaultDFF(int dModel) {
        int raw = (int) Math.round(8.0 / 3 * dModel);
        return ((raw + 63) / 64) * 64;
    }
}
