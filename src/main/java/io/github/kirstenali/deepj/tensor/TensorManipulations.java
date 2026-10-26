package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.autograd.AxisReductionOps;
import io.github.kirstenali.deepj.autograd.ViewOps;

/** Shape transformations retain one Tensor system and participate in first-order autograd. */
public abstract class TensorManipulations extends TensorStorageAccess {
    TensorManipulations(DType dtype, boolean host, int... shape) { super(dtype, host, shape); }
    TensorManipulations(TensorMemory memory, TensorLayout layout) { super(memory, layout); }
    TensorManipulations(TensorStorageAccess source, TensorLayout layout) { super(source, layout); }

    protected abstract Tensor tensor();

    public final Tensor view(int... shape) { return ViewOps.view(tensor(), shape); }
    public final Tensor unsqueeze(int axis) { return ViewOps.unsqueeze(tensor(), axis); }
    public final Tensor reshape(int... shape) { return ViewOps.reshape(tensor(), shape); }
    public final Tensor permute(int... axes) { return ViewOps.permute(tensor(), axes); }
    public final Tensor transpose(int first, int second) { return ViewOps.transpose(tensor(), first, second); }
    public final Tensor select(int axis, int index) { return ViewOps.select(tensor(), axis, index); }
    public final Tensor expand(int... shape) { return ViewOps.expand(tensor(), shape); }
    public final Tensor contiguous() { return ViewOps.contiguous(tensor()); }
    public final Tensor to(DType dtype) { return ViewOps.cast(tensor(), dtype); }

    public final Tensor slice(int axis, int start, int stop, int step) {
        return ViewOps.slice(tensor(), axis, start, stop, step);
    }

    public final Tensor slice(int axis, int start, int stop) { return slice(axis, start, stop, 1); }
    public final Tensor sum(int axis, boolean keepDimensions) {
        return AxisReductionOps.sum(tensor(), new int[]{axis}, keepDimensions);
    }
    public final Tensor sumAxes(int... axes) { return AxisReductionOps.sum(tensor(), axes, false); }
    public final Tensor softmax(int axis) { return AxisReductionOps.softmax(tensor(), axis); }
}
