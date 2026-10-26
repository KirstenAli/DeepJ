package io.github.kirstenali.deepj.tensor;

/** Axis reductions accumulate in FLOAT32 and round compact output only once per result value. */
public final class CpuNdReduction {
    private CpuNdReduction() {}

    public static Tensor sum(Tensor input, int[] axes, boolean keep) { return reduce(input, axes, keep, false); }

    public static Tensor mean(Tensor input, int[] axes, boolean keep) { return reduce(input, axes, keep, true); }

    private static Tensor reduce(Tensor input, int[] axes, boolean keep, boolean mean) {
        var shape = new ReductionShape(input.layout().shape(), axes, keep);
        Tensor output = new Tensor(input.dtype(), shape.output().dimensions());
        input.materialize();
        float[] sums = new float[output.numel()];
        for (int index = 0; index < input.numel(); index++) {
            sums[shape.outputIndex(index)] += input.memory().get(input.layout().addressAt(index));
        }
        for (int index = 0; index < sums.length; index++) {
            output.memory().set(index, mean ? sums[index] / shape.count() : sums[index]);
        }
        return output;
    }

    public static float total(Tensor input, boolean absolute) {
        input.materialize();
        float sum = 0;
        for (int index = 0; index < input.numel(); index++) {
            float value = input.memory().get(input.layout().addressAt(index));
            sum += absolute ? Math.abs(value) : value;
        }
        return sum;
    }

    public static Tensor sliceGradient(Tensor gradient, int[] dimensions, int axis, int start, int step) {
        TensorShape shape = new TensorShape(dimensions);
        axis = shape.axis(axis);
        long end = gradient.size(axis) == 0 ? start : start + (gradient.size(axis) - 1L) * step + 1;
        TensorLayout region = LayoutTransforms.slice(TensorLayout.dense(shape),
                axis, start, Math.toIntExact(end), step);
        if (!region.shape().equals(gradient.layout().shape())) {
            throw new IllegalArgumentException("Slice gradient shape must match the selected region");
        }
        gradient.materialize();
        Tensor result = new Tensor(DType.FLOAT32, dimensions);
        for (int index = 0; index < gradient.numel(); index++) {
            result.memory().set(region.addressAt(index), gradient.memory().get(gradient.layout().addressAt(index)));
        }
        return result;
    }
}
