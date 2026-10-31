package io.github.kirstenali.deepj.layers;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;

import java.util.List;
import java.util.Random;

public final class Linear implements Projection {

    private final Parameter W;
    private final Parameter b;

    public Linear(int dIn, int dOut, Random rnd) {
        this(dIn, dOut, rnd, ParameterFactory.float32());
    }

    public Linear(int dIn, int dOut, Random rnd, ParameterFactory parameters) {
        this.W = parameters.random("weight", dIn, dOut, rnd);
        this.b = parameters.zeros("bias", 1, dOut);
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
