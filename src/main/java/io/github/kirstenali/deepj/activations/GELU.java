package io.github.kirstenali.deepj.activations;

import io.github.kirstenali.deepj.tensor.Tensor;

public final class GELU implements ActivationFunction {

    private Tensor lastX;

    @Override
    public Tensor forward(Tensor input) {
        lastX = input;
        return input.geluActivation();
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        if (lastX == null) {
            throw new IllegalStateException("GELU.backward called before forward");
        }
        if (gradOutput.rows != lastX.rows || gradOutput.cols != lastX.cols) {
            throw new IllegalArgumentException("gradOutput shape must match input shape");
        }
        return lastX.geluBackward(gradOutput);
    }
}
