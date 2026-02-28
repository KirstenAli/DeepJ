package org.DeepJ.ann.layers;

import org.DeepJ.ann.Tensor;
import org.DeepJ.ann.optimisers.Parameter;

import java.util.List;
import java.util.Random;

/**
 * Fully-connected layer: y = xW + b
 * x: [n x dIn], W: [dIn x dOut], b: [1 x dOut]
 */
public final class Linear implements Layer {

    private final int dIn;
    private final int dOut;

    private final Parameter W;
    private final Parameter b;

    private Tensor lastX;

    public Linear(int dIn, int dOut, Random rnd) {
        this.dIn = dIn;
        this.dOut = dOut;
        this.W = new Parameter(Tensor.random(dIn, dOut, rnd));
        this.b = new Parameter(Tensor.zeros(1, dOut));
    }

    public Tensor forward(Tensor x) {
        if (x.cols != dIn) throw new IllegalArgumentException("Expected input cols=" + dIn + " got " + x.cols);
        this.lastX = x;
        Tensor y = x.matmul(W.value).addRowVector(b.value);
        return y;
    }

    public Tensor backward(Tensor gradY) {
        // dW = X^T * dY
        Tensor dW = lastX.transpose().matmul(gradY);
        // db = sumRows(dY)
        Tensor db = gradY.sumRows();

        // accumulate
        W.grad = W.grad.add(dW);
        b.grad = b.grad.add(db);

        // dX = dY * W^T
        return gradY.matmul(W.value.transpose());
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(W, b);
    }

    public Parameter weight() { return W; }
    public Parameter bias() { return b; }

    @Override
    public Tensor backward(Tensor gradOutput, double learningRate) {
        return backward(gradOutput);
    }

}
