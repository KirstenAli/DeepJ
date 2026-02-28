package io.github.kirstenali.deepj.transformer.embeddings;

import io.github.kirstenali.deepj.Tensor;
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
        Tensor out = new Tensor(ids.length, dModel);
        for (int i = 0; i < ids.length; i++) {
            int id = ids[i];
            if (id < 0 || id >= vocabSize) throw new IllegalArgumentException("Token id out of range: " + id);
            System.arraycopy(weight.value.data[id], 0, out.data[i], 0, dModel);
        }
        return out;
    }

    public void backward(Tensor gradOut) {
        // accumulate dW for the rows used
        for (int i = 0; i < lastIds.length; i++) {
            int id = lastIds[i];
            for (int j = 0; j < dModel; j++) {
                weight.grad.data[id][j] += gradOut.data[i][j];
            }
        }
    }

    public Parameter weight() {
        return weight;
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(weight);
    }
}
