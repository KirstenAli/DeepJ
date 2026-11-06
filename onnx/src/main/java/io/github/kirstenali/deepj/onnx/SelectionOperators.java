package io.github.kirstenali.deepj.onnx;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;

final class SelectionOperators {
    private SelectionOperators() {}

    static void register(OnnxRegistry r) {
        r.add("ArgMax", 1, "axis keepdims select_last_index", c -> List.of(arg(c, true)));
        r.add("ArgMin", 1, "axis keepdims select_last_index", c -> List.of(arg(c, false)));
        r.add("TopK", 10, "axis largest sorted", SelectionOperators::top);
    }

    private static OnnxValue arg(OnnxContext c, boolean maximum) {
        OnnxValue input = c.input(0);
        int axis = ValueOps.axis(c.integer("axis", 0), input.shape().length);
        int[] shape = input.shape();
        int size = shape[axis], inner = ValueOps.product(shape, axis + 1, shape.length);
        if (size == 0) throw new IllegalArgumentException("Arg reduction needs a nonempty axis");
        long[] output = new long[input.size() / size];
        float[] floats = input.isFloating() ? input.tensor().toFloatArray() : null;
        for (int i = 0; i < output.length; i++) {
            output[i] = best(input, floats, i, size, inner, maximum, c.integer("select_last_index", 0) != 0);
        }
        shape[axis] = 1;
        if (c.integer("keepdims", 1) == 0)
            shape = IntStream.range(0, shape.length).filter(i -> i != axis).map(i -> input.shape()[i]).toArray();
        return OnnxValue.integers(output, shape);
    }

    private static int best(OnnxValue input, float[] floats, int index, int size, int inner, boolean maximum,
            boolean last) {
        int base = index / inner * size * inner + index % inner, chosen = 0;
        for (int i = 1; i < size; i++) {
            int comparison = compare(input, floats, base + i * inner, base + chosen * inner);
            if ((maximum ? comparison > 0 : comparison < 0) || last && comparison == 0) chosen = i;
        }
        return chosen;
    }

    private static int compare(OnnxValue value, float[] floats, int a, int b) {
        return floats == null ? Long.compare(value.data()[a], value.data()[b]) : Float.compare(floats[a], floats[b]);
    }

    private static List<OnnxValue> top(OnnxContext c) {
        OnnxValue input = c.input(0);
        int[] shape = input.shape();
        int axis = ValueOps.axis(c.integer("axis", -1), shape.length), size = shape[axis];
        int count = Math.toIntExact(c.input(1).scalar());
        if (count < 0 || count > size) throw new IllegalArgumentException("Invalid TopK count");
        int inner = ValueOps.product(shape, axis + 1, shape.length), outer = ValueOps.product(shape, 0, axis);
        shape[axis] = count;
        long[] selected = new long[OnnxValue.elements(shape)];
        int[] addresses = new int[selected.length];
        float[] floats = input.isFloating() ? input.tensor().toFloatArray() : null;
        for (int i = 0; i < outer * inner; i++) {
            fillTop(c, input, floats, i, size, inner, count, selected, addresses);
        }
        return List.of(ValueOps.take(input, addresses, shape), OnnxValue.integers(selected, shape));
    }

    private static void fillTop(OnnxContext c, OnnxValue input, float[] floats, int group, int size, int inner,
            int count, long[] selected, int[] addresses) {
        int base = group / inner * size * inner + group % inner;
        Comparator<Integer> order = (a, b) -> compare(input, floats, base + a * inner, base + b * inner);
        if (c.integer("largest", 1) != 0) order = order.reversed();
        Integer[] indices = IntStream.range(0, size).boxed().toArray(Integer[]::new);
        Arrays.sort(indices, order.thenComparingInt(Integer::intValue));
        for (int i = 0; i < count; i++) {
            int output = (group / inner * count + i) * inner + group % inner;
            selected[output] = indices[i];
            addresses[output] = base + indices[i] * inner;
        }
    }
}
