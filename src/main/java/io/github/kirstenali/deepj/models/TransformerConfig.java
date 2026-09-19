package io.github.kirstenali.deepj.models;

public interface TransformerConfig {
    int vocabSize();
    int maxSeqLen();
    int dModel();
    int nHeads();
    int nLayers();
    int dFF();
    float gradClipNorm();

    static void validateCommon(int vocabSize, int maxSeqLen, int dModel,
                               int nHeads, int nLayers, int dFF, float gradClipNorm) {
        if (vocabSize <= 0) throw new IllegalArgumentException("vocabSize must be > 0");
        if (maxSeqLen <= 0) throw new IllegalArgumentException("maxSeqLen must be > 0");
        if (dModel <= 0)    throw new IllegalArgumentException("dModel must be > 0");
        if (nHeads <= 0)    throw new IllegalArgumentException("nHeads must be > 0");
        if (nLayers <= 0)   throw new IllegalArgumentException("nLayers must be > 0");
        if (dFF <= 0)       throw new IllegalArgumentException("dFF must be > 0");
        if (dModel % nHeads != 0)
            throw new IllegalArgumentException("dModel must be divisible by nHeads");
        if (!Float.isFinite(gradClipNorm) || gradClipNorm <= 0.0f)
            throw new IllegalArgumentException("gradClipNorm must be finite and > 0");
    }

    static void validateRotaryHeadDimension(int dModel, int nHeads) {
        if ((dModel / nHeads) % 2 != 0) {
            throw new IllegalArgumentException("Rotary attention requires an even head dimension");
        }
    }
}
