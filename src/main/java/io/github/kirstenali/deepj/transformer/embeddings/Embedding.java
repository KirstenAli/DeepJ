package io.github.kirstenali.deepj.transformer.embeddings;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.training.Trainable;

import java.util.List;
import java.util.Random;

public final class Embedding implements Trainable {

    private final int vocabSize;
    private final int dModel;

    private final Parameter weight;

    public Embedding(int vocabSize, int dModel, Random rnd) {
        this(vocabSize, dModel, rnd, ParameterFactory.float32());
    }

    public Embedding(int vocabSize, int dModel, Random rnd, ParameterFactory parameters) {
        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.weight = parameters.random("weight", vocabSize, dModel, rnd);
    }

    public Tensor forward(int[] ids) {
        for (int id : ids) {
            if (id < 0 || id >= vocabSize) throw new IllegalArgumentException("Token id out of range: " + id);
        }
        return Tensor.sliceRows(weight.value, ids, dModel);
    }

    public Tensor forwardBatch(int[][] sequences) {
        TokenBatch batch = TokenBatch.from(sequences);
        return forward(batch.tokens()).reshape(batch.batches(), batch.length(), dModel);
    }

    public Parameter weight() {
        return weight;
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(weight);
    }
}
