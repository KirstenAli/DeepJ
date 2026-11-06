package io.github.kirstenali.deepj.onnx;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

final class ReshapeOperators {
    private ReshapeOperators() {}

    static void register(OnnxRegistry registry) {
        registry.add("Reshape", 5, "allowzero", context -> List.of(reshape(context)));
        registry.add("Flatten", 1, "axis", context -> List.of(flatten(context)));
        registry.add("Squeeze", 1, "axes", context -> List.of(squeeze(context, false)));
        registry.add("Unsqueeze", 1, "axes", context -> List.of(squeeze(context, true)));
    }

    private static OnnxValue reshape(OnnxContext context) {
        int[] shape = reshapeDimensions(context);
        int inferred = -1, count = 1;
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] == -1) {
                if (inferred >= 0)
                    throw new IllegalArgumentException("Multiple inferred reshape dimensions");
                inferred = i;
            } else count = Math.multiplyExact(count, shape[i]);
        }
        if (inferred >= 0) {
            if (count == 0 || context.input(0).size() % count != 0)
                throw new IllegalArgumentException("Invalid inferred reshape");
            shape[inferred] = context.input(0).size() / count;
        }
        return ValueOps.reshape(context.input(0), shape);
    }

    private static OnnxValue flatten(OnnxContext context) {
        int[] shape = context.input(0).shape();
        int axis = context.integer("axis", 1);
        if (axis < 0) axis += shape.length;
        if (axis < 0 || axis > shape.length)
            throw new IllegalArgumentException("Invalid Flatten axis");
        return ValueOps.reshape(
                context.input(0),
                new int[] {
                    ValueOps.product(shape, 0, axis), ValueOps.product(shape, axis, shape.length)
                });
    }

    private static OnnxValue squeeze(OnnxContext context, boolean insert) {
        int[] original = context.input(0).shape();
        requireAxes(context, insert);
        int[] axes =
                context.axes(
                        1,
                        IntStream.range(0, original.length)
                                .filter(i -> original[i] == 1)
                                .toArray());
        int rank = original.length + (insert ? axes.length : 0);
        int[] normalized = normalizedAxes(axes, rank);
        return ValueOps.reshape(
                context.input(0), squeezedShape(original, normalized, rank, insert));
    }

    private static int[] reshapeDimensions(OnnxContext context) {
        int[] shape = context.input(1).ints(), original = context.input(0).shape();
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] < -1)
                throw new IllegalArgumentException("Invalid negative reshape dimension");
            if (shape[i] == 0 && context.integer("allowzero", 0) == 0) {
                if (i >= original.length)
                    throw new IllegalArgumentException("Reshape cannot copy missing dimension");
                shape[i] = original[i];
            }
        }
        return shape;
    }

    private static void requireSingleton(int dimension) {
        if (dimension != 1) throw new IllegalArgumentException("Squeeze dimension must be one");
    }

    private static void requireAxes(OnnxContext context, boolean insert) {
        if (insert && !context.hasInput(1) && !context.hasAttribute("axes")) {
            throw new IllegalArgumentException("Unsqueeze requires axes");
        }
    }

    private static int[] normalizedAxes(int[] axes, int rank) {
        int[] normalized = Arrays.stream(axes).map(i -> ValueOps.axis(i, rank)).sorted().toArray();
        if (Arrays.stream(normalized).distinct().count() != normalized.length) {
            throw new IllegalArgumentException("Duplicate squeeze axes");
        }
        return normalized;
    }

    private static int[] squeezedShape(int[] original, int[] axes, int rank, boolean insert) {
        var shape = new ArrayList<Integer>();
        int source = 0;
        for (int i = 0; i < rank; i++) {
            boolean selected = Arrays.binarySearch(axes, i) >= 0;
            if (insert && selected) shape.add(1);
            else if (!insert && selected) requireSingleton(original[source++]);
            else shape.add(original[source++]);
        }
        return shape.stream().mapToInt(Integer::intValue).toArray();
    }
}
