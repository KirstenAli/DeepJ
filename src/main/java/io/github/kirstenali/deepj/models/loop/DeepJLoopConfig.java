package io.github.kirstenali.deepj.models.loop;

import io.github.kirstenali.deepj.models.TransformerConfig;
import io.github.kirstenali.deepj.models.nova.DeepJNovaConfig;

/** Unique layer counts and the number of fully differentiated recurrent passes. */
public record DeepJLoopConfig(int vocabSize, int maxSeqLen, int dModel, int nHeads, int kvHeads,
                               int preludeLayers, int coreLayers, int codaLayers, int loops, int dFF,
                               boolean qkNorm, float initScale, float gradClipNorm,
                               float coreResidualScale) implements TransformerConfig {

    public DeepJLoopConfig(int vocabSize, int maxSeqLen, int dModel, int nHeads, int kvHeads,
                           int preludeLayers, int coreLayers, int codaLayers, int loops, int dFF) {
        this(vocabSize, maxSeqLen, dModel, nHeads, kvHeads, preludeLayers, coreLayers, codaLayers,
                loops, dFF, true, 0.2f, 1.0f, (float) (1.0 / (loops * Math.sqrt(coreLayers))));
    }

    public DeepJLoopConfig {
        if (preludeLayers < 0 || codaLayers < 0 || coreLayers <= 0 || loops <= 0) {
            throw new IllegalArgumentException("Outer layer counts must be >= 0; coreLayers and loops must be > 0");
        }
        int unique = Math.addExact(Math.addExact(preludeLayers, coreLayers), codaLayers);
        Math.addExact(Math.addExact(preludeLayers, Math.multiplyExact(coreLayers, loops)), codaLayers);
        new DeepJNovaConfig(vocabSize, maxSeqLen, dModel, nHeads, kvHeads, unique, dFF,
                qkNorm, initScale, gradClipNorm);
        if (!Float.isFinite(coreResidualScale) || coreResidualScale <= 0.0f) {
            throw new IllegalArgumentException("coreResidualScale must be finite and > 0");
        }
    }

    @Override
    public int nLayers() {
        return preludeLayers + coreLayers + codaLayers;
    }

    public int executedLayers() {
        return preludeLayers + coreLayers * loops + codaLayers;
    }

    /** Attention and MLP dimensions shared with the existing Nova implementation. */
    public DeepJNovaConfig backbone() {
        return new DeepJNovaConfig(vocabSize, maxSeqLen, dModel, nHeads, kvHeads, nLayers(),
                dFF, qkNorm, initScale, gradClipNorm);
    }
}
