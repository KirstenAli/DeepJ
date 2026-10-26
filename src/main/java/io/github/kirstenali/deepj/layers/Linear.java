package io.github.kirstenali.deepj.layers;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.optimisers.Parameter;

import java.util.List;
import java.util.Random;

public final class Linear implements Projection {

    private final Parameter W;
    private final Parameter b;

    public Linear(int dIn, int dOut, Random rnd) {
        this.W = new Parameter(Tensor.random(dIn, dOut, rnd));
        this.b = new Parameter(Tensor.zeros(1, dOut));
    }

    @Override
    public Tensor forward(Tensor x) {
        return ProjectionOps.affine(x, W.value, b.value);
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(W, b);
    }

    public Parameter weight() { return W; }
    public Parameter bias() { return b; }
}
