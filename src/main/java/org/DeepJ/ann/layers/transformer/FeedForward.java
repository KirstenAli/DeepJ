package org.DeepJ.ann.layers.transformer;

import org.DeepJ.ann.Tensor;
import org.DeepJ.ann.activations.ActivationFunction;
import org.DeepJ.ann.activations.GELU;
import org.DeepJ.ann.layers.Layer;
import org.DeepJ.ann.layers.Linear;
import org.DeepJ.ann.optimisers.Parameter;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Transformer feed-forward network (FFN / MLP):
 * Linear(dModel->dFF) -> activation -> Linear(dFF->dModel)
 */
public final class FeedForward implements Layer {

    private final Linear fc1;
    private final Linear fc2;
    private final ActivationFunction activation;

    public FeedForward(int dModel, int dFF, Random rnd) {
        this(dModel, dFF, new GELU(), rnd);
    }

    public FeedForward(int dModel, int dFF, ActivationFunction activation, Random rnd) {
        if (dModel <= 0) throw new IllegalArgumentException("dModel must be > 0");
        if (dFF <= 0) throw new IllegalArgumentException("dFF must be > 0");
        if (activation == null) throw new IllegalArgumentException("activation must not be null");
        if (rnd == null) throw new IllegalArgumentException("rnd must not be null");

        this.fc1 = new Linear(dModel, dFF, rnd);
        this.fc2 = new Linear(dFF, dModel, rnd);
        this.activation = activation;
    }

    @Override
    public Tensor forward(Tensor x) {
        Tensor h1 = fc1.forward(x);
        Tensor h2 = activation.forward(h1);
        return fc2.forward(h2);
    }

    public Tensor backward(Tensor gradOut) {
        Tensor g2 = fc2.backward(gradOut);
        Tensor g1 = activation.backward(g2);
        return fc1.backward(g1);
    }

    @Override
    public Tensor backward(Tensor gradOutput, double learningRate) {
        return backward(gradOutput);
    }

    @Override
    public List<Parameter> parameters() {
        List<Parameter> ps = new ArrayList<>();
        ps.addAll(fc1.parameters());
        ps.addAll(fc2.parameters());
        return ps;
    }
}
