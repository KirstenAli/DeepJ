package io.github.kirstenali.deepj.optimisers;

import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.Arrays;

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
        if (canReuseCpuGradBuffer()) {
            clearGradData();
            return;
        }
        this.grad = Tensor.zeros(value.rows, value.cols);
    }

    private boolean canReuseCpuGradBuffer() {
        return grad != null
                && grad.rows == value.rows
                && grad.cols == value.cols
                && grad.getGpuTag() == null;
    }

    private void clearGradData() {
        Arrays.fill(grad.data, 0.0);
    }
}
