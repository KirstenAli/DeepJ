package io.github.kirstenali.deepj.nn;

/** Pre-normalized encoder dimensions. Input embeddings supply their chosen positional representation. */
public record TransformerEncoderConfig(int width, int heads, int hiddenSize, int layers, float dropout) {
    public TransformerEncoderConfig {
        NeuralChecks.positive(width, "width");
        NeuralChecks.positive(heads, "heads");
        NeuralChecks.positive(hiddenSize, "feed-forward size");
        NeuralChecks.positive(layers, "encoder depth");
        NeuralChecks.probability(dropout, "dropout");
        if (width % heads != 0) throw new IllegalArgumentException("Width must be divisible by heads");
    }
}
