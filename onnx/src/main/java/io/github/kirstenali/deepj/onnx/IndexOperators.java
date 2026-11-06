package io.github.kirstenali.deepj.onnx;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.stream.IntStream;

final class IndexOperators {
    private IndexOperators() {}

    static void register(OnnxRegistry registry) {
        registry.add("Slice", 1, "starts ends axes steps", context -> List.of(slice(context)));
        registry.add("Gather", 1, "axis", context -> List.of(gather(context)));
        registry.add("GatherElements", 11, "axis", context -> List.of(gatherElements(context)));
        registry.add("GatherND", 11, "batch_dims", context -> List.of(gatherNd(context)));
    }

    static OnnxValue select(OnnxValue value, int axis, int[] indices) {
        int[] shape = value.shape();
        shape[axis] = indices.length;
        if (value.isFloating()) return OnnxValue.of(value.tensor().indexSelect(axis, indices));
        int inner = ValueOps.product(shape, axis + 1, shape.length);
        int[] addresses = new int[OnnxValue.elements(shape)];
        for (int i = 0; i < addresses.length; i++) {
            int outer = i / inner / indices.length, position = i / inner % indices.length;
            addresses[i] = (outer * value.shape()[axis] + indices[position]) * inner + i % inner;
        }
        return ValueOps.take(value, addresses, shape);
    }

    private static OnnxValue gather(OnnxContext context) {
        OnnxValue input = context.input(0), index = context.input(1);
        int[] shape = input.shape();
        int axis = ValueOps.axis(context.integer("axis", 0), shape.length);
        int[] indices =
                Arrays.stream(index.data()).mapToInt(i -> normalize(i, shape[axis])).toArray();
        int[] output =
                ValueOps.join(
                        Arrays.copyOfRange(shape, 0, axis),
                        index.shape(),
                        Arrays.copyOfRange(shape, axis + 1, shape.length));
        return ValueOps.reshape(select(input, axis, indices), output);
    }

    private static OnnxValue gatherElements(OnnxContext context) {
        int[] data = context.input(0).shape(), shape = context.input(1).shape();
        if (shape.length != data.length)
            throw new IllegalArgumentException("GatherElements ranks differ");
        int axis = ValueOps.axis(context.integer("axis", 0), shape.length);
        for (int i = 0; i < shape.length; i++) {
            if (i != axis && shape[i] > data[i])
                throw new IllegalArgumentException("GatherElements index dimension exceeds input");
        }
        int[] indices = new int[context.input(1).size()];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = elementAddress(i, context.input(1).data()[i], axis, shape, data);
        }
        return ValueOps.take(context.input(0), indices, shape);
    }

    private static int elementAddress(
            int linear, long selected, int axis, int[] shape, int[] data) {
        int address = 0, stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            int coordinate = i == axis ? normalize(selected, data[i]) : linear % shape[i];
            linear /= shape[i];
            address += coordinate * stride;
            stride *= data[i];
        }
        return address;
    }

    private static OnnxValue gatherNd(OnnxContext context) {
        int[] data = context.input(0).shape(), indices = context.input(1).shape();
        int batches = context.integer("batch_dims", 0);
        int tuple = tupleWidth(data, indices, batches);
        checkBatches(data, indices, batches);
        int[] output =
                ValueOps.join(
                        Arrays.copyOf(indices, indices.length - 1),
                        Arrays.copyOfRange(data, batches + tuple, data.length));
        int[] addresses = new int[OnnxValue.elements(output)];
        for (int i = 0; i < addresses.length; i++) {
            addresses[i] = ndAddress(i, data, indices, context.input(1).data(), batches, tuple);
        }
        return ValueOps.take(context.input(0), addresses, output);
    }

    private static int ndAddress(
            int index, int[] data, int[] indices, long[] values, int batches, int tuple) {
        int inner = ValueOps.product(data, batches + tuple, data.length), group = index / inner;
        int batchSize = ValueOps.product(indices, batches, indices.length - 1);
        int address = group / batchSize;
        for (int i = 0; i < tuple; i++) {
            address =
                    address * data[batches + i]
                            + normalize(values[group * tuple + i], data[batches + i]);
        }
        return address * inner + index % inner;
    }

    private static OnnxValue slice(OnnxContext context) {
        long[] starts = longs(context, 1, "starts"), ends = longs(context, 2, "ends");
        int[] axes = sliceAxes(context, starts.length);
        long[] steps = steps(context, starts.length);
        requireSliceLengths(starts.length, ends.length, axes.length, steps.length);
        OnnxValue result = context.input(0);
        var seen = new HashSet<Integer>();
        for (int i = 0; i < starts.length; i++) {
            int axis = ValueOps.axis(axes[i], result.shape().length);
            if (!seen.add(axis)) throw new IllegalArgumentException("Duplicate Slice axis");
            result =
                    select(
                            result,
                            axis,
                            sliceIndices(result.shape()[axis], starts[i], ends[i], steps[i]));
        }
        return result;
    }

    private static long[] longs(OnnxContext context, int input, String attribute) {
        return context.hasInput(input)
                ? context.input(input).data()
                : context.attributes.get(attribute).getIntsList().stream()
                        .mapToLong(Long::longValue)
                        .toArray();
    }

    private static int[] sliceIndices(int size, long first, long last, long step) {
        if (step == 0) throw new IllegalArgumentException("Slice step cannot be zero");
        long start = clip(first, size, step > 0), end = clip(last, size, step > 0);
        long distance = step > 0 ? end - start : start - end;
        long stride = step == Long.MIN_VALUE ? Long.MAX_VALUE : Math.abs(step);
        int count = distance <= 0 ? 0 : Math.toIntExact(1 + (distance - 1) / stride);
        int[] output = new int[count];
        for (int i = 0; i < count; i++) {
            output[i] = Math.toIntExact(start + i * step);
        }
        return output;
    }

    private static long clip(long value, int size, boolean forward) {
        long shifted = value < 0 ? Math.max(value, -(long) size - 1) + size : value;
        return Math.max(forward ? 0 : -1, Math.min(forward ? size : size - 1L, shifted));
    }

    static int normalize(long index, int size) {
        long result = index < 0 ? index + size : index;
        if (result < 0 || result >= size)
            throw new IllegalArgumentException("ONNX index out of range: " + index);
        return Math.toIntExact(result);
    }

    private static void checkBatches(int[] data, int[] indices, int batches) {
        for (int i = 0; i < batches; i++) {
            if (data[i] != indices[i])
                throw new IllegalArgumentException("GatherND batches differ");
        }
    }

    private static long[] steps(OnnxContext context, int count) {
        if (context.hasInput(4)) return context.input(4).data();
        return Arrays.stream(
                        context.integers("steps", IntStream.range(0, count).map(i -> 1).toArray()))
                .asLongStream()
                .toArray();
    }

    private static int tupleWidth(int[] data, int[] indices, int batches) {
        if (indices.length == 0
                || batches < 0
                || batches >= Math.min(data.length, indices.length)) {
            throw new IllegalArgumentException("Invalid GatherND batch dimensions");
        }
        int tuple = indices[indices.length - 1];
        if (tuple < 1 || tuple + batches > data.length)
            throw new IllegalArgumentException("Invalid GatherND tuple width");
        return tuple;
    }

    private static int[] sliceAxes(OnnxContext context, int count) {
        return context.hasInput(3)
                ? context.input(3).ints()
                : context.integers("axes", IntStream.range(0, count).toArray());
    }

    private static void requireSliceLengths(int starts, int ends, int axes, int steps) {
        if (starts != ends || starts != axes || starts != steps) {
            throw new IllegalArgumentException("Slice argument lengths differ");
        }
    }
}
