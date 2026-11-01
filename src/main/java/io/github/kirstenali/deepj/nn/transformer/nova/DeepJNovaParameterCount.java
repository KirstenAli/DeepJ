package io.github.kirstenali.deepj.nn.transformer.nova;

public final class DeepJNovaParameterCount {

    private DeepJNovaParameterCount() {}

    public static long count(DeepJNovaConfig config) {
        long width = config.dModel(), headWidth = width / config.nHeads();
        long attention = 2L * width * width + 2L * width * config.kvHeads() * headWidth;
        long feedForward = 3L * width * config.dFF();
        long normalizations = 2L * width + (config.qkNorm() ? 2L * headWidth : 0L);
        return config.vocabSize() * width + config.nLayers() * (attention + feedForward + normalizations)
                + width;
    }
}
