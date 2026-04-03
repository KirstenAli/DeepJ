package io.github.kirstenali.deepj.transformer.embeddings;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.training.Trainable;

import java.util.List;
import java.util.Random;

/**
 * Learnable positional embeddings added to token embeddings.
 */
public final class PositionalEmbedding implements Trainable {

    private final int maxSeq;
    private final int dModel;
    private final Parameter weight; // [maxSeq x dModel]
    private int lastSeqLen;

    public PositionalEmbedding(int maxSeq, int dModel, Random rnd) {
        this.maxSeq = maxSeq;
        this.dModel = dModel;
        this.weight = new Parameter(Tensor.random(maxSeq, dModel, rnd));
    }

    public Tensor forward(int seqLen) {
        if (seqLen > maxSeq) throw new IllegalArgumentException("seqLen " + seqLen + " exceeds maxSeq " + maxSeq);
        this.lastSeqLen = seqLen;

        int[] indices = new int[seqLen];
        for (int i = 0; i < seqLen; i++) indices[i] = i;
        return Tensor.sliceRows(weight.value, indices, dModel);
    }

    public void backward(Tensor gradOut) {
        int seqLen = gradOut.rows;
        int[] indices = new int[seqLen];
        for (int i = 0; i < seqLen; i++) indices[i] = i;
        Tensor.scatterAddRows(weight.grad, indices, gradOut);
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(weight);
    }
}
