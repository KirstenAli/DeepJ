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
        requirePositive(vocabSize, "vocabSize");
        requirePositive(maxSeqLen, "maxSeqLen");
        requirePositive(dModel, "dModel");
        requirePositive(nHeads, "nHeads");
        requirePositive(nLayers, "nLayers");
        requirePositive(dFF, "dFF");
        requireDivisibleModelWidth(dModel, nHeads);
        requireValidGradientClip(gradClipNorm);
    }

    private static void requirePositive(int value, String name) {
        if (value <= 0) throw new IllegalArgumentException(name + " must be > 0");
    }

    private static void requireDivisibleModelWidth(int dModel, int nHeads) {
        if (dModel % nHeads != 0) throw new IllegalArgumentException("dModel must be divisible by nHeads");
    }

    private static void requireValidGradientClip(float gradClipNorm) {
        if (!Float.isFinite(gradClipNorm) || gradClipNorm <= 0.0f) {
            throw new IllegalArgumentException("gradClipNorm must be finite and > 0");
        }
    }

    static void validateRotaryHeadDimension(int dModel, int nHeads) {
        if ((dModel / nHeads) % 2 != 0) {
            throw new IllegalArgumentException("Rotary attention requires an even head dimension");
        }
    }
}
