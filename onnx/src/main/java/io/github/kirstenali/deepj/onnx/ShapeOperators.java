package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

final class ShapeOperators {
    private ShapeOperators() {}

    static void register(OnnxRegistry registry) {
        registerConstants(registry);
        registry.add("Shape", 1, "start end", ShapeOperators::shape);
        registry.add("Size", 1, "", context -> List.of(ValueOps.scalar(context.input(0).size())));
        registry.add("Transpose", 1, "perm", context -> List.of(transpose(context)));
        registry.add(
                "Expand",
                8,
                "",
                context -> List.of(ValueOps.expand(context.input(0), context.input(1).ints())));
        registry.add(
                "Concat",
                4,
                "axis",
                context ->
                        List.of(ValueOps.concatenate(inputs(context), context.integer("axis", 0))));
        registry.add("Split", 2, "axis split num_outputs", ShapeOperators::split);
        adapters(registry);
    }

    private static List<OnnxValue> inputs(OnnxContext context) {
        return IntStream.range(0, context.inputCount()).mapToObj(context::input).toList();
    }

    private static List<OnnxValue> constant(OnnxContext context) {
        if (context.hasAttribute("value"))
            return List.of(context.session.constant(context.tensorAttribute("value")));
        if (context.hasAttribute("value_int"))
            return List.of(ValueOps.scalar(context.longAttribute("value_int", 0)));
        if (context.hasAttribute("value_ints")) return constantIntegers(context);
        if (context.hasAttribute("value_float"))
            return context.one(
                    Tensor.from(new float[] {context.floating("value_float", 0)}, new int[0]));
        var values = context.attributes.get("value_floats").getFloatsList();
        float[] output = new float[values.size()];
        for (int i = 0; i < output.length; i++) {
            output[i] = values.get(i);
        }
        return context.one(Tensor.from(output, output.length));
    }

    private static List<OnnxValue> constantShape(OnnxContext context) {
        int[] shape = context.input(0).ints();
        OnnxValue scalar =
                context.hasAttribute("value")
                        ? context.session.constant(context.tensorAttribute("value"))
                        : OnnxValue.of(Tensor.from(new float[] {0}, new int[0]));
        if (scalar.size() != 1)
            throw new IllegalArgumentException("ConstantOfShape value must have one element");
        return List.of(ValueOps.expand(ValueOps.reshape(scalar, new int[0]), shape));
    }

    private static List<OnnxValue> shape(OnnxContext context) {
        int[] shape = context.input(0).shape();
        int start = bound(context.longAttribute("start", 0), shape.length),
                end = bound(context.longAttribute("end", shape.length), shape.length);
        long[] output = Arrays.stream(shape, start, Math.max(start, end)).asLongStream().toArray();
        return List.of(OnnxValue.integers(output, output.length));
    }

    private static int bound(long index, int rank) {
        return (int) Math.max(0, Math.min(rank, index < 0 ? index + rank : index));
    }

    private static OnnxValue transpose(OnnxContext context) {
        int[] shape = context.input(0).shape();
        int[] axes =
                context.integers(
                        "perm",
                        IntStream.range(0, shape.length).map(i -> shape.length - i - 1).toArray());
        if (axes.length != shape.length
                || Arrays.stream(axes).distinct().count() != axes.length
                || Arrays.stream(axes).anyMatch(a -> a < 0 || a >= shape.length)) {
            throw new IllegalArgumentException("Invalid ONNX transpose permutation");
        }
        int[] output =
                Arrays.stream(axes).map(i -> shape[ValueOps.axis(i, shape.length)]).toArray();
        if (context.input(0).isFloating()) return OnnxValue.of(context.tensor(0).permute(axes));
        int[] indices = new int[OnnxValue.elements(output)];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = transposeIndex(i, output, shape, axes);
        }
        return ValueOps.take(context.input(0), indices, output);
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

    private static List<OnnxValue> split(OnnxContext context) {
        OnnxValue input = context.input(0);
        int axis = ValueOps.axis(context.integer("axis", 0), input.shape().length),
                length = input.shape()[axis];
        int[] sizes = context.hasInput(1) ? context.input(1).ints() : context.integers("split");
        if (sizes.length == 0) sizes = splitSizes(context, length);
        if (sizes.length != context.outputCount()
                || Arrays.stream(sizes).asLongStream().sum() != length
                || Arrays.stream(sizes).anyMatch(s -> s < 0))
            throw new IllegalArgumentException("Invalid Split lengths");
        var output = new ArrayList<OnnxValue>();
        int start = 0;
        for (int size : sizes) {
            int end = Math.addExact(start, size);
            output.add(IndexOperators.select(input, axis, IntStream.range(start, end).toArray()));
            start += size;
        }
        return output;
    }

    private static int[] splitSizes(OnnxContext context, int length) {
        int count = context.integer("num_outputs", context.outputCount());
        if (count != context.outputCount() || count <= 0)
            throw new IllegalArgumentException("Invalid Split output count");
        if (!context.hasAttribute("num_outputs") && length % count != 0)
            throw new IllegalArgumentException("Unequal implicit Split");
        int[] sizes = new int[count];
        Arrays.fill(sizes, (length + count - 1) / count);
        sizes[count - 1] = length - sizes[0] * (count - 1);
        return sizes;
    }

    private static OnnxValue tile(OnnxContext context) {
        int[] shape = context.input(0).shape(), repeats = context.input(1).ints();
        if (repeats.length != shape.length || Arrays.stream(repeats).anyMatch(r -> r < 0))
            throw new IllegalArgumentException("Tile repeats rank differs");
        int[] output =
                IntStream.range(0, shape.length)
                        .map(i -> Math.multiplyExact(shape[i], repeats[i]))
                        .toArray();
        int[] indices = new int[OnnxValue.elements(output)];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = tileIndex(i, output, shape);
        }
        return ValueOps.take(context.input(0), indices, output);
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

    private static void adapters(OnnxRegistry registry) {
        ReshapeOperators.register(registry);
        IndexOperators.register(registry);
        CastOperators.register(registry);
        registry.add("Tile", 6, "", context -> List.of(tile(context)));
    }

    private static void registerConstants(OnnxRegistry registry) {
        registry.add("Identity", 1, "", context -> List.of(context.input(0)));
        registry.add(
                "Constant",
                9,
                "value value_float value_int value_floats value_ints",
                ShapeOperators::constant);
        registry.add("ConstantOfShape", 9, "value", ShapeOperators::constantShape);
    }

    private static List<OnnxValue> constantIntegers(OnnxContext context) {
        long[] values =
                context.attributes.get("value_ints").getIntsList().stream()
                        .mapToLong(Long::longValue)
                        .toArray();
        return List.of(OnnxValue.integers(values, values.length));
    }
}
