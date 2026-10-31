package io.github.kirstenali.deepj.tensor;

/** Stable logical-axis softmax without an intermediate compact-dtype exponential tensor. */
public final class CpuNdSoftmax {
    private CpuNdSoftmax() {}

    public static Tensor apply(Tensor input, int axis) {
        axis = input.layout().shape().axis(axis);
        Tensor output = new Tensor(input.dtype(), input.shape());
        if (input.numel() == 0) return output;
        input.materialize();
        int inner = Math.toIntExact(input.layout().shape().denseStrides()[axis]);
        int width = input.size(axis);
        for (int group = 0; group < input.numel() / width; group++) {
            int base = group / inner * width * inner + group % inner;
            row(input, output, base, width, inner);
        }
        return output;
    }

    private static void row(Tensor input, Tensor output, int base, int width, int stride) {
        float max = Float.NEGATIVE_INFINITY;
        for (int index = 0; index < width; index++) {
            max = Math.max(max, value(input, base + index * stride));
        }
        float sum = 0;
        for (int index = 0; index < width; index++) {
            sum += exponential(value(input, base + index * stride) - max);
        }
        for (int index = 0; index < width; index++) {
            float probability = exponential(value(input, base + index * stride) - max) / sum;
            output.memory().set(base + index * stride, probability);
        }
    }

    private static float value(Tensor input, int logical) {
        return input.memory().get(input.layout().addressAt(logical));
    }

    private static float exponential(float value) { return (float) Math.exp(value); }
}
