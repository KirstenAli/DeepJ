package io.github.kirstenali.deepj.models.gpt;

public record GPTConfig(
        int vocabSize,
        int maxSeqLen,
        int dModel,
        int nHeads,
        int nLayers,
        int dFF
) {}
