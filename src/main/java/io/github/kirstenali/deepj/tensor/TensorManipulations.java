package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.autograd.AxisReductionOps;
import io.github.kirstenali.deepj.autograd.ViewOps;
import io.github.kirstenali.deepj.autograd.LayoutOps;

/** Shape transformations retain one Tensor system and participate in first-order autograd. */
public abstract class TensorManipulations extends TensorStorageAccess {
    public static Tensor zeros(DType dtype, int... shape) { return new Tensor(dtype, shape); }
    public static Tensor ones(DType dtype, int... shape) { return TensorInitializers.ones(dtype, shape); }
    public static Tensor random(DType dtype, java.util.Random random, int... shape) {
        return TensorInitializers.random(dtype, random, shape);
    }

    TensorManipulations(DType dtype, boolean host, int... shape) { super(dtype, host, shape); }
    TensorManipulations(DType dtype, boolean host, TensorLayout geometry) { super(dtype, host, geometry); }
    TensorManipulations(TensorMemory memory, TensorLayout layout) { super(memory, layout); }
    TensorManipulations(TensorStorageAccess source, TensorLayout layout) { super(source, layout); }

    protected abstract Tensor tensor();

    public boolean requiresGrad() { return gradientState().required(); }

    public final Tensor view(int... shape) { return ViewOps.view(tensor(), shape); }
    public final Tensor unsqueeze(int axis) { return ViewOps.unsqueeze(tensor(), axis); }
    public final Tensor reshape(int... shape) { return ViewOps.reshape(tensor(), shape); }
    public final Tensor permute(int... axes) { return ViewOps.permute(tensor(), axes); }
    public final Tensor transpose(int first, int second) { return ViewOps.transpose(tensor(), first, second); }
    public final Tensor select(int axis, int index) { return ViewOps.select(tensor(), axis, index); }
    public final Tensor expand(int... shape) { return ViewOps.expand(tensor(), shape); }
    public final Tensor contiguous() { return ViewOps.contiguous(tensor()); }
    public final Tensor to(DType dtype) { return ViewOps.cast(tensor(), dtype); }
    /** Independent value assignment with an explicit cast to this tensor's dtype; shapes must match. */
    public final Tensor copyFrom(Tensor source) { return TensorMutation.assign(tensor(), source); }

    public final Tensor slice(int axis, int start, int stop, int step) {
        return ViewOps.slice(tensor(), axis, start, stop, step);
    }

    public final Tensor slice(int axis, int start, int stop) { return slice(axis, start, stop, 1); }
    public final Tensor sum(int axis, boolean keepDimensions) {
        return AxisReductionOps.sum(tensor(), new int[]{axis}, keepDimensions);
    }
    public final Tensor sumAxes(int... axes) { return AxisReductionOps.sum(tensor(), axes, false); }
    public final Tensor mean(int axis, boolean keep) { return AxisReductionOps.mean(tensor(), new int[]{axis}, keep); }
    public final Tensor meanAxes(int... axes) { return AxisReductionOps.mean(tensor(), axes, false); }
    public final Tensor variance(int axis, boolean keep, int correction) {
        return AxisReductionOps.variance(tensor(), new int[]{axis}, keep, correction);
    }
    /** Population variance; use the correction overload for a different denominator. */
    public final Tensor variance(int axis, boolean keep) { return variance(axis, keep, 0); }
    public final Tensor varianceAxes(int[] axes, boolean keep, int correction) {
        return AxisReductionOps.variance(tensor(), axes, keep, correction);
    }
    /** Tied maxima share their gradient equally. */
    public final Tensor maximum(int axis, boolean keep) {
        return AxisReductionOps.maximum(tensor(), new int[]{axis}, keep);
    }
    public final Tensor maxAxes(int... axes) { return AxisReductionOps.maximum(tensor(), axes, false); }
    public final Tensor softmax(int axis) { return AxisReductionOps.softmax(tensor(), axis); }

    public final Tensor indexSelect(int axis, int... indices) { return LayoutOps.indexSelect(tensor(), axis, indices); }

    public final Tensor indexAddInPlace(int axis, int[] indices, Tensor updates) {
        return TensorMutation.indexAdd(tensor(), axis, indices, updates);
    }
}
