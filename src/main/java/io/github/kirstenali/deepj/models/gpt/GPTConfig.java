package io.github.kirstenali.deepj.models.gpt;

import io.github.kirstenali.deepj.models.TransformerConfig;

public record GPTConfig(
        int vocabSize,
        int maxSeqLen,
        int dModel,
        int nHeads,
        int nLayers,
        int dFF,
        double initScale,
        double gradClipNorm
) implements TransformerConfig {

    public GPTConfig(int vocabSize, int maxSeqLen, int dModel, int nHeads, int nLayers, int dFF) {
        this(vocabSize, maxSeqLen, dModel, nHeads, nLayers, dFF, 0.2, 1.0);
    }

    public GPTConfig {
        TransformerConfig.validateCommon(vocabSize, maxSeqLen, dModel, nHeads, nLayers, dFF, gradClipNorm);
        if (!Double.isFinite(initScale) || initScale <= 0.0)
            throw new IllegalArgumentException("initScale must be finite and > 0");
    }
}
