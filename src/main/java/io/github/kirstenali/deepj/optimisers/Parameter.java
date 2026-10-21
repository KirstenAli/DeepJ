package io.github.kirstenali.deepj.optimisers;

import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.Arrays;

public final class Parameter {
    public final Tensor value;

    public Parameter(Tensor value) {
        this.value = value.retainDeviceBuffer().requiresGrad(true);
        zeroGrad();
    }

    public Tensor grad() { return value.grad(); }

    public void setGradient(Tensor gradient) {
        if (gradient != null && gradient.requiresGrad()) {
            throw new IllegalArgumentException("Parameter gradients must be detached");
        }
        if (gradient != null) Tensor.requireSameShape(value, gradient, "parameter gradient");
        value.gradientState().bindGradient(gradient);
    }

    public void zeroGrad() {
        if (!value.requiresGrad()) {
            setGradient(null);
            return;
        }
        if (canReuseCpuGradBuffer()) {
            clearGradData();
            return;
        }
        setGradient(Tensor.zeros(value.rows, value.cols));
    }

    private boolean canReuseCpuGradBuffer() {
        return grad() != null && grad().getGpuTag() == null;
    }

    private void clearGradData() {
        grad().gradientState().beforeMutation();
        Arrays.fill(grad().data, 0.0f);
    }
}
