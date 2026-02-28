package io.github.kirstenali.deepj.optimisers;

import io.github.kirstenali.deepj.Tensor;

/**
 * Simple mutable parameter holder for optimizers.
 * Gradients are expected to be accumulated into {@link #grad}.
 */
public final class Parameter {
    public Tensor value;
    public Tensor grad;

    public Parameter(Tensor value) {
        this.value = value;
        this.grad = Tensor.zeros(value.rows, value.cols);
    }

    public void zeroGrad() {
        this.grad = Tensor.zeros(value.rows, value.cols);
    }
}
