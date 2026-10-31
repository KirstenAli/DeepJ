package io.github.kirstenali.deepj.transformer.embeddings;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.training.Trainable;

import java.util.List;
import java.util.Random;

public final class PositionalEmbedding implements Trainable {

    private final int maxSeq;
    private final int dModel;
    private final Parameter weight;

    public PositionalEmbedding(int maxSeq, int dModel, Random rnd) {
        this(maxSeq, dModel, rnd, ParameterFactory.float32());
    }

    public PositionalEmbedding(int maxSeq, int dModel, Random rnd, ParameterFactory parameters) {
        this.maxSeq = maxSeq;
        this.dModel = dModel;
        this.weight = parameters.random("weight", maxSeq, dModel, rnd);
    }

    public Tensor forward(int seqLen) {
        return forward(seqLen, 0);
    }

    public Tensor forward(int seqLen, int offset) {
        if (seqLen < 1 || offset < 0 || (long) seqLen + offset > maxSeq) {
            throw new IllegalArgumentException("Position range exceeds maxSeq " + maxSeq);
        }
        int[] indices = new int[seqLen];
        for (int i = 0; i < seqLen; i++) {
            indices[i] = offset + i;
        }
        return Tensor.sliceRows(weight.value, indices, dModel);
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(weight);
    }
}
