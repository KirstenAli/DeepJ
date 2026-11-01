package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.Arrays;

/** Flattens an inclusive axis range; defaults to preserving the first (batch) axis. */
public final class Flatten extends Module {
    private final int first;
    private final int last;

    public Flatten() { this(1, -1); }
    public Flatten(int first, int last) {
        this.first = first;
        this.last = last;
    }

    @Override public Tensor forward(Tensor input) {
        int start = axis(input, first), end = axis(input, last);
        if (start > end) throw new IllegalArgumentException("Flatten axes must be ordered");
        int[] original = input.shape(), shape = Arrays.copyOf(original, original.length - end + start);
        shape[start] = 1;
        for (int index = start; index <= end; index++) {
            shape[start] = Math.multiplyExact(shape[start], original[index]);
        }
        System.arraycopy(original, end + 1, shape, start + 1, original.length - end - 1);
        return input.reshape(shape);
    }

    private static int axis(Tensor input, int axis) {
        input.size(axis);
        return axis < 0 ? input.rank() + axis : axis;
    }
}
