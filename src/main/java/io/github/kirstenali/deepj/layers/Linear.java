package io.github.kirstenali.deepj.layers;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.optimisers.Parameter;

import java.util.List;
import java.util.Random;

public final class Linear implements Projection {

    private final int dIn;

    private final Parameter W;
    private final Parameter b;

    public Linear(int dIn, int dOut, Random rnd) {
        this.dIn = dIn;
        this.W = new Parameter(Tensor.random(dIn, dOut, rnd));
        this.b = new Parameter(Tensor.zeros(1, dOut));
    }

    @Override
    public Tensor forward(Tensor x) {
        if (x.cols != dIn) throw new IllegalArgumentException("Expected input cols=" + dIn + " got " + x.cols);
        Tensor y = x.matmul(W.value).addRowVector(b.value);
        return y;
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(W, b);
    }

    public Parameter weight() { return W; }
    public Parameter bias() { return b; }
}
