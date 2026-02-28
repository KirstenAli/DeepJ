package org.DeepJ.ann.transformer.embeddings;

import org.DeepJ.ann.Tensor;
import org.DeepJ.ann.optimisers.Parameter;
import org.DeepJ.ann.training.Trainable;

import java.util.List;
import java.util.Random;

/**
 * Learnable positional embeddings added to token embeddings.
 */
public final class PositionalEmbedding implements Trainable {

    private final int maxSeq;
    private final int dModel;
    private final Parameter weight; // [maxSeq x dModel]

    public PositionalEmbedding(int maxSeq, int dModel, Random rnd) {
        this.maxSeq = maxSeq;
        this.dModel = dModel;
        this.weight = new Parameter(Tensor.random(maxSeq, dModel, rnd));
    }

    public Tensor forward(int seqLen) {
        if (seqLen > maxSeq) throw new IllegalArgumentException("seqLen " + seqLen + " exceeds maxSeq " + maxSeq);
        Tensor out = new Tensor(seqLen, dModel);
        for (int i = 0; i < seqLen; i++) {
            System.arraycopy(weight.value.data[i], 0, out.data[i], 0, dModel);
        }
        return out;
    }

    public void backward(Tensor gradOut) {
        // accumulate only first seqLen rows
        for (int i = 0; i < gradOut.rows; i++) {
            for (int j = 0; j < dModel; j++) {
                weight.grad.data[i][j] += gradOut.data[i][j];
            }
        }
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(weight);
    }
}
