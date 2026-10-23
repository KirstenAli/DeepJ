package io.github.kirstenali.deepj.activations;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.autograd.ActivationOps;

public final class Softmax implements ActivationFunction {
    @Override
    public Tensor forward(Tensor input) {
        return input.softmaxRows();
    }

    public Tensor forwardCausal(Tensor logits, int sequenceLength, float scale) {
        return ActivationOps.causalSoftmax(logits, sequenceLength, scale);
    }
}
