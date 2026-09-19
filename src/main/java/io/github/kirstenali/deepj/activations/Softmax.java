package io.github.kirstenali.deepj.activations;

import io.github.kirstenali.deepj.tensor.Tensor;

public final class Softmax implements ActivationFunction {

    private Tensor softmaxOut;

    @Override
    public Tensor forward(Tensor logits) {
        this.softmaxOut = logits.softmaxRows();
        return softmaxOut;
    }

    public Tensor forwardCausal(Tensor logits, int sequenceLength, float scale) {
        this.softmaxOut = Tensor.backend().causalSoftmax(logits, sequenceLength, scale);
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
