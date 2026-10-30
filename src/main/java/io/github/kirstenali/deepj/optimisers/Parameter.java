package io.github.kirstenali.deepj.optimisers;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.DType;

import java.util.Arrays;
import java.util.Objects;

public final class Parameter {
    public final Tensor value;
    private final String name;

    public Parameter(Tensor value) {
        this("", value);
    }

    public Parameter(String name, Tensor value) {
        this.name = Objects.requireNonNull(name, "name");
        this.value = value.retainDeviceBuffer().requiresGrad(true);
        zeroGrad();
    }

    public String name() { return name; }

    public Tensor grad() { return value.grad(); }

    public void setGradient(Tensor gradient) {
        if (gradient != null && gradient.requiresGrad()) {
            throw new IllegalArgumentException("Parameter gradients must be detached");
        }
        if (gradient != null && gradient.dtype() != DType.FLOAT32) {
            throw new IllegalArgumentException("Parameter gradients must accumulate in FLOAT32");
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
        setGradient(Tensor.backend().zeros(DType.FLOAT32, value.shape()));
    }

    private boolean canReuseCpuGradBuffer() {
        return grad() != null && grad().getGpuTag() == null && grad().isContiguous()
                && grad().storageOffset() == 0 && grad().storageBytes() == (long) grad().numel() * Float.BYTES;
    }

    private void clearGradData() {
        grad().gradientState().beforeMutation();
        if (grad().rawData() != null) Arrays.fill(grad().rawData(), 0.0f);
    }
}
