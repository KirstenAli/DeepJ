package io.github.kirstenali.deepj.tensor;

import java.util.Arrays;

final class ReductionShape {
    private final TensorShape input, output;
    private final boolean[] reduced;
    private final long count;

    ReductionShape(TensorShape input, int[] axes, boolean keepDimensions) {
        this.input = input;
        reduced = axes(input, axes);
        output = output(input, reduced, keepDimensions);
        long size = 1;
        for (int axis = 0; axis < input.rank(); axis++) {
            if (reduced[axis]) size = Math.multiplyExact(size, input.size(axis));
        }
        count = size;
    }

    TensorShape output() { return output; }
    long count() { return count; }

    int outputIndex(int index) {
        int result = 0, stride = 1;
        for (int axis = input.rank() - 1; axis >= 0; axis--) {
            int coordinate = index % input.size(axis);
            index /= input.size(axis);
            if (!reduced[axis]) {
                result += coordinate * stride;
                stride *= input.size(axis);
            }
        }
        return result;
    }

    private static boolean[] axes(TensorShape input, int[] axes) {
        boolean[] result = new boolean[input.rank()];
        if (axes.length == 0) Arrays.fill(result, true);
        for (int axis : axes) {
            int normalized = input.axis(axis);
            if (result[normalized]) throw new IllegalArgumentException("Reduction axes cannot repeat");
            result[normalized] = true;
        }
        return result;
    }

    private static TensorShape output(TensorShape input, boolean[] reduced, boolean keep) {
        int[] dimensions = input.dimensions();
        int count = 0;
        for (int axis = 0; axis < dimensions.length; axis++) {
            if (!reduced[axis]) dimensions[count++] = input.size(axis);
            else if (keep) dimensions[count++] = 1;
        }
        return new TensorShape(Arrays.copyOf(dimensions, count));
    }
}
