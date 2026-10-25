package io.github.kirstenali.deepj.tensor;

import java.util.Objects;

final class ShapeTransforms {
    private ShapeTransforms() {}

    static TensorShape broadcast(TensorShape left, TensorShape right) {
        int[] output = new int[Math.max(left.rank(), right.rank())];
        for (int axis = output.length - 1; axis >= 0; axis--) {
            output[axis] = dimension(trailing(left, axis, output.length), trailing(right, axis, output.length));
        }
        return new TensorShape(output);
    }

    private static int trailing(TensorShape shape, int axis, int rank) {
        int source = axis - rank + shape.rank();
        return source < 0 ? 1 : shape.size(source);
    }

    private static int dimension(int left, int right) {
        if (left == right || right == 1) return left;
        if (left == 1) return right;
        throw new IllegalArgumentException("Dimensions do not broadcast: " + left + " and " + right);
    }

    static TensorShape reshape(TensorShape source, int... requested) {
        int[] dimensions = Objects.requireNonNull(requested, "requested").clone();
        int inferred = inferredAxis(dimensions);
        if (inferred >= 0) infer(source.elements(), dimensions, inferred);
        TensorShape result = new TensorShape(dimensions);
        if (source.elements() != result.elements()) throw new IllegalArgumentException("Reshape changes element count");
        return result;
    }

    private static int inferredAxis(int[] dimensions) {
        int inferred = -1;
        for (int axis = 0; axis < dimensions.length; axis++) {
            if (dimensions[axis] < -1) throw new IllegalArgumentException("Invalid reshape dimension");
            if (dimensions[axis] != -1) continue;
            if (inferred >= 0) throw new IllegalArgumentException("Only one reshape dimension can be inferred");
            inferred = axis;
        }
        return inferred;
    }

    private static void infer(int elements, int[] dimensions, int axis) {
        dimensions[axis] = 1;
        int known = new TensorShape(dimensions).elements();
        if (known == 0) throw new IllegalArgumentException("Cannot infer a dimension from a zero-size product");
        if (elements % known != 0) throw new IllegalArgumentException("Reshape dimensions do not divide element count");
        dimensions[axis] = elements / known;
    }
}
