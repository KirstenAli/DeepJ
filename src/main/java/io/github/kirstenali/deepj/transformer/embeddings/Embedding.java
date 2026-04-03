package io.github.kirstenali.deepj.transformer.embeddings;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.training.Trainable;

import java.util.List;
import java.util.Random;

/**
 * Token embedding: ids -> vectors. Input ids are provided via {@link #forward(int[])}.
 * Output is a tensor of shape [nTokens x dModel].
 */
public final class Embedding implements Trainable {

    private final int vocabSize;
    private final int dModel;

    private final Parameter weight; // [vocabSize x dModel]
    private int[] lastIds;

    public Embedding(int vocabSize, int dModel, Random rnd) {
        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.weight = new Parameter(Tensor.random(vocabSize, dModel, rnd));
    }

    public Tensor forward(int[] ids) {
        this.lastIds = ids;
        for (int id : ids) {
            if (id < 0 || id >= vocabSize) throw new IllegalArgumentException("Token id out of range: " + id);
        }
        return Tensor.sliceRows(weight.value, ids, dModel);
    }

    public void backward(Tensor gradOut) {
        Tensor.scatterAddRows(weight.grad, lastIds, gradOut);
    }

    public Parameter weight() {
        return weight;
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(weight);
    }
}
