package io.github.kirstenali.deepj.activations;

import io.github.kirstenali.deepj.tensor.Tensor;

/**
 * Row-wise softmax for 2D tensors: applies softmax independently to each row.
 *
 * Forward:  softmaxRows(logits)
 * Backward: given upstream grad dY, returns dLogits using cached softmax output.
 */
public final class Softmax implements ActivationFunction {

    private Tensor softmaxOut; // cache from forward

    @Override
    public Tensor forward(Tensor logits) {
        this.softmaxOut = logits.softmaxRows();
        return softmaxOut;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        if (softmaxOut == null) {
            throw new IllegalStateException("SoftmaxRows.backward() called before forward()");
        }
        Tensor.requireSameShape(gradOutput, softmaxOut, "SoftmaxRows.backward");
        return gradOutput.softmaxBackward(softmaxOut);
    }
}