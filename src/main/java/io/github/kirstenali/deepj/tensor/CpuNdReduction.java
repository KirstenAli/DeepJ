package io.github.kirstenali.deepj.tensor;

/** Axis reductions accumulate in FLOAT32 and round compact output only once per result value. */
public final class CpuNdReduction {
    private CpuNdReduction() {}

    public static Tensor sum(Tensor input, int[] axes, boolean keep) { return reduce(input, axes, keep, false); }

    public static Tensor mean(Tensor input, int[] axes, boolean keep) { return reduce(input, axes, keep, true); }

    private static Tensor reduce(Tensor input, int[] axes, boolean keep, boolean mean) {
        var shape = new ReductionShape(input.layout().shape(), axes, keep);
        Tensor output = new Tensor(input.dtype(), shape.output().dimensions());
        float[] sums = sums(input, shape);
        for (int index = 0; index < sums.length; index++) {
            output.memory().set(index, mean ? sums[index] / shape.count() : sums[index]);
        }
        return output;
    }

    static float[] sums(Tensor input, ReductionShape shape) {
        input.materialize();
        float[] sums = new float[shape.output().elements()];
        for (int index = 0; index < input.numel(); index++) {
            sums[shape.outputIndex(index)] += input.memory().get(input.layout().addressAt(index));
        }
        return sums;
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
        TensorLayout region = SliceGradientLayout.of(gradient, dimensions, axis, start, step);
        gradient.materialize();
        Tensor result = new Tensor(DType.FLOAT32, dimensions);
        for (int index = 0; index < gradient.numel(); index++) {
            result.memory().set(region.addressAt(index), gradient.memory().get(gradient.layout().addressAt(index)));
        }
        return result;
    }
}
