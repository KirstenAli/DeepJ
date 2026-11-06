package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

final class ShapeOperators {
    private ShapeOperators() {}

    static void register(OnnxRegistry r) {
        r.add("Identity", 1, "", c -> List.of(c.input(0)));
        r.add("Constant", 9, "value value_float value_int value_floats value_ints", ShapeOperators::constant);
        r.add("ConstantOfShape", 9, "value", ShapeOperators::constantShape);
        r.add("Shape", 1, "start end", ShapeOperators::shape);
        r.add("Size", 1, "", c -> List.of(ValueOps.scalar(c.input(0).size())));
        r.add("Reshape", 5, "allowzero", c -> List.of(reshape(c)));
        r.add("Flatten", 1, "axis", c -> List.of(flatten(c)));
        r.add("Transpose", 1, "perm", c -> List.of(transpose(c)));
        r.add("Expand", 8, "", c -> List.of(ValueOps.expand(c.input(0), c.input(1).ints())));
        r.add("Concat", 4, "axis", c -> List.of(ValueOps.concatenate(inputs(c), c.integer("axis", 0))));
        r.add("Split", 2, "axis split num_outputs", ShapeOperators::split);
        r.add("Squeeze", 1, "axes", c -> List.of(squeeze(c, false)));
        r.add("Unsqueeze", 1, "axes", c -> List.of(squeeze(c, true)));
        adapters(r);
    }

    private static List<OnnxValue> inputs(OnnxContext c) {
        return IntStream.range(0, c.inputCount()).mapToObj(c::input).toList();
    }

    private static List<OnnxValue> constant(OnnxContext c) {
        if (c.hasAttribute("value")) return List.of(c.session.constant(c.tensorAttribute("value")));
        if (c.hasAttribute("value_int"))
            return List.of(ValueOps.scalar(c.longAttribute("value_int", 0)));
        if (c.hasAttribute("value_ints")) {
            long[] values = c.attributes.get("value_ints").getIntsList().stream().mapToLong(Long::longValue).toArray();
            return List.of(OnnxValue.integers(values, values.length));
        }
        if (c.hasAttribute("value_float"))
            return c.one(Tensor.from(new float[] {c.floating("value_float", 0)}, new int[0]));
        var values = c.attributes.get("value_floats").getFloatsList();
        float[] output = new float[values.size()];
        for (int i = 0; i < output.length; i++) {
            output[i] = values.get(i);
        }
        return c.one(Tensor.from(output, output.length));
    }

    private static List<OnnxValue> constantShape(OnnxContext c) {
        int[] shape = c.input(0).ints();
        OnnxValue scalar = c.hasAttribute("value") ? c.session.constant(c.tensorAttribute("value"))
                        : OnnxValue.of(Tensor.from(new float[] {0}, new int[0]));
        if (scalar.size() != 1)
            throw new IllegalArgumentException("ConstantOfShape value must have one element");
        return List.of(ValueOps.expand(ValueOps.reshape(scalar, new int[0]), shape));
    }

    private static List<OnnxValue> shape(OnnxContext c) {
        int[] shape = c.input(0).shape();
        int start = bound(c.longAttribute("start", 0), shape.length),
                end = bound(c.longAttribute("end", shape.length), shape.length);
        long[] output = Arrays.stream(shape, start, Math.max(start, end)).asLongStream().toArray();
        return List.of(OnnxValue.integers(output, output.length));
    }

    private static int bound(long index, int rank) {
        return (int) Math.max(0, Math.min(rank, index < 0 ? index + rank : index));
    }

    private static OnnxValue reshape(OnnxContext c) {
        int[] shape = reshapeDimensions(c);
        int inferred = -1, count = 1;
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] == -1) {
                if (inferred >= 0)
                    throw new IllegalArgumentException("Multiple inferred reshape dimensions");
                inferred = i;
            } else count = Math.multiplyExact(count, shape[i]);
        }
        if (inferred >= 0) {
            if (count == 0 || c.input(0).size() % count != 0)
                throw new IllegalArgumentException("Invalid inferred reshape");
            shape[inferred] = c.input(0).size() / count;
        }
        return ValueOps.reshape(c.input(0), shape);
    }

    private static OnnxValue flatten(OnnxContext c) {
        int[] shape = c.input(0).shape();
        int axis = c.integer("axis", 1);
        if (axis < 0) axis += shape.length;
        if (axis < 0 || axis > shape.length)
            throw new IllegalArgumentException("Invalid Flatten axis");
        return ValueOps.reshape(c.input(0), new int[] {
                    ValueOps.product(shape, 0, axis), ValueOps.product(shape, axis, shape.length)
                });
    }

    private static OnnxValue transpose(OnnxContext c) {
        int[] shape = c.input(0).shape();
        int[] axes = c.integers("perm", IntStream.range(0, shape.length).map(i -> shape.length - i - 1).toArray());
        if (axes.length != shape.length
                || Arrays.stream(axes).distinct().count() != axes.length
                || Arrays.stream(axes).anyMatch(a -> a < 0 || a >= shape.length)) {
            throw new IllegalArgumentException("Invalid ONNX transpose permutation");
        }
        int[] output = Arrays.stream(axes).map(i -> shape[ValueOps.axis(i, shape.length)]).toArray();
        if (c.input(0).isFloating()) return OnnxValue.of(c.tensor(0).permute(axes));
        int[] indices = new int[OnnxValue.elements(output)];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = transposeIndex(i, output, shape, axes);
        }
        return ValueOps.take(c.input(0), indices, output);
    }

    private static int transposeIndex(int index, int[] output, int[] input, int[] axes) {
        int[] coordinates = new int[input.length];
        for (int i = output.length - 1; i >= 0; i--) {
            coordinates[axes[i]] = index % output[i];
            index /= output[i];
        }
        int address = 0;
        for (int i = 0; i < input.length; i++) {
            address = address * input[i] + coordinates[i];
        }
        return address;
    }

    private static List<OnnxValue> split(OnnxContext c) {
        OnnxValue input = c.input(0);
        int axis = ValueOps.axis(c.integer("axis", 0), input.shape().length), length = input.shape()[axis];
        int[] sizes = c.hasInput(1) ? c.input(1).ints() : c.integers("split");
        if (sizes.length == 0) sizes = splitSizes(c, length);
        if (sizes.length != c.outputCount()
                || Arrays.stream(sizes).asLongStream().sum() != length
                || Arrays.stream(sizes).anyMatch(s -> s < 0))
            throw new IllegalArgumentException("Invalid Split lengths");
        var output = new ArrayList<OnnxValue>();
        int start = 0;
        for (int size : sizes) {
            output.add(IndexOperators.select(input, axis,
                            IntStream.range(start, Math.addExact(start, size)).toArray()));
            start += size;
        }
        return output;
    }

    private static int[] splitSizes(OnnxContext c, int length) {
        int count = c.integer("num_outputs", c.outputCount());
        if (count != c.outputCount() || count <= 0)
            throw new IllegalArgumentException("Invalid Split output count");
        if (!c.hasAttribute("num_outputs") && length % count != 0)
            throw new IllegalArgumentException("Unequal implicit Split");
        int[] sizes = new int[count];
        Arrays.fill(sizes, (length + count - 1) / count);
        sizes[count - 1] = length - sizes[0] * (count - 1);
        return sizes;
    }

    private static OnnxValue squeeze(OnnxContext c, boolean insert) {
        int[] original = c.input(0).shape();
        requireAxes(c, insert);
        int[] axes = c.axes(1, IntStream.range(0, original.length).filter(i -> original[i] == 1).toArray());
        int rank = original.length + (insert ? axes.length : 0);
        int[] normalized = Arrays.stream(axes).map(i -> ValueOps.axis(i, rank)).sorted().toArray();
        if (Arrays.stream(normalized).distinct().count() != normalized.length)
            throw new IllegalArgumentException("Duplicate squeeze axes");
        var shape = new ArrayList<Integer>();
        int source = 0;
        for (int i = 0; i < rank; i++) {
            boolean selected = Arrays.binarySearch(normalized, i) >= 0;
            if (insert && selected) shape.add(1);
            else if (!insert && selected) requireSingleton(original[source++]);
            else shape.add(original[source++]);
        }
        return ValueOps.reshape(c.input(0), shape.stream().mapToInt(Integer::intValue).toArray());
    }

    private static OnnxValue tile(OnnxContext c) {
        int[] shape = c.input(0).shape(), repeats = c.input(1).ints();
        if (repeats.length != shape.length || Arrays.stream(repeats).anyMatch(r -> r < 0))
            throw new IllegalArgumentException("Tile repeats rank differs");
        int[] output = IntStream.range(0, shape.length).map(i -> Math.multiplyExact(shape[i], repeats[i])).toArray();
        int[] indices = new int[OnnxValue.elements(output)];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = tileIndex(i, output, shape);
        }
        return ValueOps.take(c.input(0), indices, output);
    }

    private static int tileIndex(int index, int[] output, int[] shape) {
        int address = 0, stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            address += index % output[i] % shape[i] * stride;
            index /= output[i];
            stride *= shape[i];
        }
        return address;
    }

    private static void adapters(OnnxRegistry r) {
        IndexOperators.register(r);
        CastOperators.register(r);
        r.add("Tile", 6, "", c -> List.of(tile(c)));
    }

    private static int[] reshapeDimensions(OnnxContext c) {
        int[] shape = c.input(1).ints(), original = c.input(0).shape();
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] < -1)
                throw new IllegalArgumentException("Invalid negative reshape dimension");
            if (shape[i] == 0 && c.integer("allowzero", 0) == 0) {
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

    private static void requireAxes(OnnxContext c, boolean insert) {
        if (insert && !c.hasInput(1) && !c.hasAttribute("axes")) {
            throw new IllegalArgumentException("Unsqueeze requires axes");
        }
    }
}
