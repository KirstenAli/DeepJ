package io.github.kirstenali.deepj.models.orbit;

import io.github.kirstenali.deepj.models.TransformerConfig;

public record DeepJOrbitConfig(
        int vocabSize,
        int maxSeqLen,
        int dModel,
        int nHeads,
        int nLayers,
        int dFF,
        float initScale,
        float gradClipNorm
) implements TransformerConfig {

    public DeepJOrbitConfig(int vocabSize, int maxSeqLen, int dModel, int nHeads, int nLayers, int dFF) {
        this(vocabSize, maxSeqLen, dModel, nHeads, nLayers, dFF, 0.2f, 1.0f);
    }

    public DeepJOrbitConfig(int vocabSize, int maxSeqLen, int dModel, int nHeads, int nLayers,
                       int dFF, float gradClipNorm) {
        this(vocabSize, maxSeqLen, dModel, nHeads, nLayers, dFF, 0.2f, gradClipNorm);
    }

    public DeepJOrbitConfig {
        TransformerConfig.validateCommon(vocabSize, maxSeqLen, dModel, nHeads, nLayers, dFF, gradClipNorm);
        TransformerConfig.validateRotaryHeadDimension(dModel, nHeads);
        if (!Float.isFinite(initScale) || initScale <= 0.0f) {
            throw new IllegalArgumentException("initScale must be finite and > 0");
        }
    }

    public static int defaultDFF(int dModel) {
        int raw = (int) Math.round(8.0 / 3 * dModel);
        return ((raw + 63) / 64) * 64;
    }
}
