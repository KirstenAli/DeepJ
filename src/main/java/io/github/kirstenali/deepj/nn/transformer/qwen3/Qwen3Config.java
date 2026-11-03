package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.nn.transformer.TransformerConfig;

/** Dense, bias-free Qwen3 with tied embeddings and the released default RoPE configuration. */
public record Qwen3Config(int vocabSize, int maxSeqLen, int dModel, int nHeads, int kvHeads,
                          int nLayers, int dFF, int headDim, double ropeTheta,
                          float initializerRange, float gradClipNorm) implements TransformerConfig {
    public Qwen3Config {
        TransformerConfig.validateCommon(vocabSize, maxSeqLen, dModel, nHeads, nLayers, dFF, gradClipNorm);
        validateHeads(nHeads, kvHeads, headDim);
        if (!(ropeTheta > 0) || !Double.isFinite(ropeTheta)
                || !(initializerRange > 0) || !Float.isFinite(initializerRange)) {
            throw new IllegalArgumentException("Invalid rotary base or initialization scale");
        }
        Math.multiplyExact(nHeads, headDim);
        Math.multiplyExact(kvHeads, headDim);
    }

    private static void validateHeads(int heads, int keys, int width) {
        if (keys < 1 || heads % keys != 0 || width < 2 || width % 2 != 0) {
            throw new IllegalArgumentException("Invalid grouped-query head geometry");
        }
    }

    public static Qwen3Config onePointSevenB() {
        return new Qwen3Config(151936, 40960, 2048, 16, 8, 28, 6144, 128, 1_000_000, 0.02f, 1);
    }

    public float rmsNormEpsilon() { return 1e-6f; }
    @Override public float initScale() { return initializerRange / 0.1f; }

    public long parameterCount() {
        long attention = 2L * dModel * headDim * (nHeads + (long) kvHeads);
        long block = attention + 3L * dModel * dFF + 2L * dModel + 2L * headDim;
        return Math.addExact(Math.multiplyExact(block, nLayers), (long) vocabSize * dModel + dModel);
    }
}
