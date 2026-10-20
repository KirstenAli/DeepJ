package io.github.kirstenali.deepj.models.nova;

import io.github.kirstenali.deepj.models.TransformerConfig;

public record DeepJNovaConfig(int vocabSize, int maxSeqLen, int dModel, int nHeads,
                              int kvHeads, int nLayers, int dFF, boolean qkNorm,
                              float initScale, float gradClipNorm) implements TransformerConfig {

    public DeepJNovaConfig(int vocabSize, int maxSeqLen, int dModel, int nHeads,
                           int kvHeads, int nLayers, int dFF) {
        this(vocabSize, maxSeqLen, dModel, nHeads, kvHeads, nLayers, dFF, true, 0.2f, 1.0f);
    }

    public DeepJNovaConfig {
        TransformerConfig.validateCommon(vocabSize, maxSeqLen, dModel, nHeads, nLayers, dFF, gradClipNorm);
        TransformerConfig.validateRotaryHeadDimension(dModel, nHeads);
        if (kvHeads <= 0 || nHeads % kvHeads != 0) {
            throw new IllegalArgumentException("nHeads must be divisible by positive kvHeads");
        }
        if (!Float.isFinite(initScale) || initScale <= 0.0f) {
            throw new IllegalArgumentException("initScale must be finite and > 0");
        }
    }
}
