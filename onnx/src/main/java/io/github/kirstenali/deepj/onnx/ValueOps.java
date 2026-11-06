package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.nn.TensorJoin;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.function.LongBinaryOperator;

final class ValueOps {
    private ValueOps() {}

    static int axis(int axis, int rank) {
        int result = axis < 0 ? axis + rank : axis;
        if (result < 0 || result >= rank)
            throw new IllegalArgumentException("Invalid ONNX axis " + axis);
        return result;
    }

    static OnnxValue reshape(OnnxValue value, int[] shape) {
        if (OnnxValue.elements(shape) != value.size())
            throw new IllegalArgumentException("ONNX reshape changes element count");
        return value.isFloating() ? OnnxValue.of(value.tensor().reshape(shape))
                : OnnxValue.host(value.data(), shape, value.elementType());
    }

    static OnnxValue take(OnnxValue value, int[] indices, int[] shape) {
        if (value.isFloating())
            return OnnxValue.of(value.tensor().reshape(value.size()).indexSelect(0, indices).reshape(shape));
        long[] output = new long[indices.length];
        for (int i = 0; i < indices.length; i++) {
            output[i] = value.data()[indices[i]];
        }
        return OnnxValue.host(output, shape, value.elementType());
    }

    static int[] broadcast(int[] a, int[] b) {
        int rank = Math.max(a.length, b.length), result[] = new int[rank];
        for (int i = 0; i < rank; i++) {
            int x = dimension(a, rank, i), y = dimension(b, rank, i);
            if (x != y && x != 1 && y != 1)
                throw new IllegalArgumentException("ONNX broadcast dimensions differ");
            result[i] = x == 1 ? y : x;
        }
        return result;
    }

    static int dimension(int[] shape, int rank, int axis) {
        int index = axis - rank + shape.length;
        return index < 0 ? 1 : shape[index];
    }

    static int address(int linear, int[] output, int[] source) {
        int address = 0, stride = 1;
        for (int axis = output.length - 1; axis >= 0; axis--) {
            int coordinate = output[axis] == 0 ? 0 : linear % output[axis];
            linear = output[axis] == 0 ? 0 : linear / output[axis];
            int size = dimension(source, output.length, axis);
            address += (size == 1 ? 0 : coordinate) * stride;
            stride *= size;
        }
        return address;
    }

    static OnnxValue expand(OnnxValue value, int[] shape) {
        int[] target = broadcast(value.shape(), shape);
        if (value.isFloating()) return OnnxValue.of(value.tensor().expand(target));
        int[] indices = new int[OnnxValue.elements(target)];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = address(i, target, value.shape());
        }
        return take(value, indices, target);
    }

    static OnnxValue binary(OnnxValue a, OnnxValue b, LongBinaryOperator operation) {
        sameType(a, b);
        int[] shape = broadcast(a.shape(), b.shape());
        long[] output = new long[OnnxValue.elements(shape)];
        for (int i = 0; i < output.length; i++) {
            output[i] = operation.applyAsLong(a.data()[address(i, shape, a.shape())],
                            b.data()[address(i, shape, b.shape())]);
        }
        return OnnxValue.host(output, shape, a.elementType());
    }

    static void sameType(OnnxValue a, OnnxValue b) {
        if (a.elementType() != b.elementType())
            throw new IllegalArgumentException("ONNX input element types differ");
    }

    static OnnxValue concatenate(List<OnnxValue> inputs, int requested) {
        OnnxValue first = inputs.get(0);
        int axis = axis(requested, first.shape().length), shape[] = first.shape();
        shape[axis] = 0;
        for (var input : inputs) {
            sameType(first, input);
            validateJoin(first.shape(), input.shape(), axis);
            shape[axis] = Math.addExact(shape[axis], input.shape()[axis]);
        }
        if (first.isFloating())
            return OnnxValue.of(TensorJoin.concatenate(
                            axis, inputs.stream().map(OnnxValue::tensor).toArray(Tensor[]::new)));
        return concatenateHost(inputs, axis, shape);
    }

    private static void validateJoin(int[] first, int[] shape, int axis) {
        if (first.length != shape.length)
            throw new IllegalArgumentException("ONNX concat ranks differ");
        for (int i = 0; i < shape.length; i++) {
            if (i != axis && first[i] != shape[i])
                throw new IllegalArgumentException("ONNX concat dimensions differ");
        }
    }

    private static OnnxValue concatenateHost(List<OnnxValue> inputs, int axis, int[] shape) {
        int inner = product(shape, axis + 1, shape.length), outer = product(shape, 0, axis);
        long[] output = new long[OnnxValue.elements(shape)];
        int offset = 0;
        for (int batch = 0; batch < outer; batch++) {
            for (var input : inputs) {
                int count = input.shape()[axis] * inner;
                System.arraycopy(input.data(), batch * count, output, offset, count);
                offset += count;
            }
        }
        return OnnxValue.host(output, shape, inputs.get(0).elementType());
    }

    static int product(int[] shape, int start, int stop) {
        return OnnxValue.elements(Arrays.copyOfRange(shape, start, stop));
    }

    static int[] join(int[]... shapes) {
        return Arrays.stream(shapes).flatMapToInt(Arrays::stream).toArray();
    }

    static OnnxValue scalar(long value) {
        return OnnxValue.integers(new long[] {value}, new int[0]);
    }

    static OnnxValue bool(boolean value) {
        return OnnxValue.booleans(new boolean[] {value}, new int[0]);
    }
}
