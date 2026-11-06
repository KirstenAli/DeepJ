package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.List;
import java.util.stream.IntStream;

final class PaddingOperators {
    private PaddingOperators() {}

    static OnnxValue pad(OnnxContext c) {
        OnnxValue value = c.input(0);
        int[] pads = c.hasInput(1) ? c.input(1).ints() : c.integers("pads");
        int[] axes = c.hasInput(3) ? c.input(3).ints() : IntStream.range(0, value.shape().length).toArray();
        if (pads.length != axes.length * 2)
            throw new IllegalArgumentException("Pad lengths differ from axes");
        OnnxValue fill = fill(c);
        var seen = new java.util.HashSet<Integer>();
        for (int i = 0; i < axes.length; i++) {
            int axis = ValueOps.axis(axes[i], value.shape().length);
            if (!seen.add(axis)) throw new IllegalArgumentException("Duplicate Pad axis");
            value = axis(value, axis, pads[i], pads[i + axes.length], fill, c.string("mode", "constant"));
        }
        return value;
    }

    private static OnnxValue fill(OnnxContext c) {
        if (c.hasInput(2)) {
            ValueOps.sameType(c.input(0), c.input(2));
            if (c.input(2).size() != 1)
                throw new IllegalArgumentException("Pad constant must be scalar");
            return ValueOps.reshape(c.input(2), new int[0]);
        }
        if (!c.input(0).isFloating())
            return OnnxValue.host(new long[] {0}, new int[0], c.input(0).elementType());
        return OnnxValue.of(Tensor.from(new float[] {c.floating("value", 0)}, c.tensor(0).dtype(), new int[0]));
    }

    private static OnnxValue axis(OnnxValue input, int axis, int before, int after, OnnxValue fill, String mode) {
        if (!java.util.Set.of("constant", "edge", "reflect", "wrap").contains(mode)) {
            throw new IllegalArgumentException("Unsupported Pad mode " + mode);
        }
        int start = Math.max(0, -before), stop = input.shape()[axis] + Math.min(0, after);
        if (start > stop) throw new IllegalArgumentException("Pad cropping exceeds the input");
        input = IndexOperators.select(input, axis, IntStream.range(start, stop).toArray());
        before = Math.max(0, before);
        after = Math.max(0, after);
        int size = input.shape()[axis], count = Math.addExact(size, Math.addExact(before, after));
        if (count < 0) throw new IllegalArgumentException("Pad cropping exceeds the input");
        boolean constant = mode.equals("constant");
        OnnxValue source = constant ? prepend(input, axis, fill) : input;
        int[] indices = new int[count];
        for (int i = 0; i < count; i++) {
            indices[i] = coordinate(i - before, size, mode) + (constant ? 1 : 0);
        }
        return IndexOperators.select(source, axis, indices);
    }

    private static OnnxValue prepend(OnnxValue input, int axis, OnnxValue fill) {
        int[] shape = input.shape();
        shape[axis] = 1;
        return ValueOps.concatenate(List.of(ValueOps.expand(fill, shape), input), axis);
    }

    private static int coordinate(int index, int size, String mode) {
        if (index >= 0 && index < size) return index;
        if (mode.equals("constant")) return -1;
        if (size == 0)
            throw new IllegalArgumentException("Cannot extend an empty axis with " + mode);
        return switch (mode) {
            case "edge" -> Math.max(0, Math.min(size - 1, index));
            case "wrap" -> Math.floorMod(index, size);
            case "reflect" -> reflect(index, size);
            default -> throw new IllegalArgumentException("Unsupported Pad mode " + mode);
        };
    }

    private static int reflect(int index, int size) {
        if (size == 1) return 0;
        int cycle = Math.multiplyExact(size - 1, 2), position = Math.floorMod(index, cycle);
        return position < size ? position : cycle - position;
    }
}
