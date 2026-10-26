package io.github.kirstenali.deepj.tensor.cpu;

import io.github.kirstenali.deepj.tensor.*;

public final class CpuBackend extends CpuTrainingOps {
    @Override public Tensor zeros(DType dtype, int... shape) { return new Tensor(dtype, shape); }

    @Override public Tensor cast(Tensor input, DType dtype) { return CpuNdElementwise.cast(input, dtype); }

    @Override public Tensor sumAxes(Tensor input, int[] axes, boolean keepDimensions) {
        return CpuNdReduction.sum(input, axes, keepDimensions);
    }

    @Override public Tensor sliceGradient(Tensor gradient, int[] shape, int axis, int start, int step) {
        return CpuNdReduction.sliceGradient(gradient, shape, axis, start, step);
    }

    @Override public Tensor softmaxAxis(Tensor input, int axis) { return CpuNdSoftmax.apply(input, axis); }
}
