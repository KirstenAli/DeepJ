package io.github.kirstenali.deepj.models;

/**
 * Common configuration fields shared by all decoder-only transformer models.
 *
 * <p>Implemented by {@link io.github.kirstenali.deepj.models.gpt.GPTConfig} and
 * {@link io.github.kirstenali.deepj.models.llama.LlamaConfig}.
 */
public interface TransformerConfig {
    int vocabSize();
    int maxSeqLen();
    int dModel();
    int nHeads();
    int nLayers();
    int dFF();
    double gradClipNorm();

    /** Validates the fields common to all transformer configs. Call from each record's compact constructor. */
    static void validateCommon(int vocabSize, int maxSeqLen, int dModel,
                               int nHeads, int nLayers, int dFF, double gradClipNorm) {
        if (vocabSize <= 0) throw new IllegalArgumentException("vocabSize must be > 0");
        if (maxSeqLen <= 0) throw new IllegalArgumentException("maxSeqLen must be > 0");
        if (dModel <= 0)    throw new IllegalArgumentException("dModel must be > 0");
        if (nHeads <= 0)    throw new IllegalArgumentException("nHeads must be > 0");
        if (nLayers <= 0)   throw new IllegalArgumentException("nLayers must be > 0");
        if (dFF <= 0)       throw new IllegalArgumentException("dFF must be > 0");
        if (dModel % nHeads != 0)
            throw new IllegalArgumentException("dModel must be divisible by nHeads");
        if (!Double.isFinite(gradClipNorm) || gradClipNorm <= 0.0)
            throw new IllegalArgumentException("gradClipNorm must be finite and > 0");
    }
}
