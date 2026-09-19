package io.github.kirstenali.deepj.models.deepseek;

public final class DeepSeekParameterCount {

    private DeepSeekParameterCount() {}

    public static long count(DeepSeekConfig config) {
        long embeddings = 2L * config.vocabSize() * config.dModel();
        long outputBias = config.vocabSize();
        long finalNorm = config.dModel();
        return embeddings + outputBias + finalNorm
                + (long) config.nLayers() * block(config);
    }

    private static long block(DeepSeekConfig config) {
        long attention = attention(config);
        long feedForward = 3L * config.dModel() * config.dFF() + 2L * config.dFF();
        long normsAndBias = 3L * config.dModel();
        return attention + feedForward + normsAndBias;
    }

    private static long attention(DeepSeekConfig config) {
        long query = 2L * config.dModel() * config.qRank();
        long keyValue = 3L * config.dModel() * config.kvRank();
        long output = (long) config.dModel() * config.dModel();
        return query + keyValue + output;
    }
}
