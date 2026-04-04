package io.github.kirstenali.deepj.models.gpt;

public record GPTConfig(
        int vocabSize,
        int maxSeqLen,
        int dModel,
        int nHeads,
        int nLayers,
        int dFF,
        double initScale,
        double gradClipNorm
) {

    public GPTConfig(int vocabSize, int maxSeqLen, int dModel, int nHeads, int nLayers, int dFF) {
        this(vocabSize, maxSeqLen, dModel, nHeads, nLayers, dFF, 0.2, 1.0);
    }

    public GPTConfig {
        if (vocabSize <= 0) {
            throw new IllegalArgumentException("vocabSize must be > 0");
        }
        if (maxSeqLen <= 0) {
            throw new IllegalArgumentException("maxSeqLen must be > 0");
        }
        if (dModel <= 0) {
            throw new IllegalArgumentException("dModel must be > 0");
        }
        if (nHeads <= 0) {
            throw new IllegalArgumentException("nHeads must be > 0");
        }
        if (nLayers <= 0) {
            throw new IllegalArgumentException("nLayers must be > 0");
        }
        if (dFF <= 0) {
            throw new IllegalArgumentException("dFF must be > 0");
        }
        if (dModel % nHeads != 0) {
            throw new IllegalArgumentException("dModel must be divisible by nHeads");
        }
        if (!Double.isFinite(initScale) || initScale <= 0.0) {
            throw new IllegalArgumentException("initScale must be finite and > 0");
        }
        if (!Double.isFinite(gradClipNorm) || gradClipNorm <= 0.0) {
            throw new IllegalArgumentException("gradClipNorm must be finite and > 0");
        }
    }
}
