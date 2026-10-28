package io.github.kirstenali.deepj.tensor;

import java.util.Arrays;
import java.util.Objects;

/** CPU logical-layout kernels with real compact storage and FLOAT32 arithmetic. */
public final class CpuNdElementwise {
    @FunctionalInterface public interface Unary { float apply(float value); }
    @FunctionalInterface public interface Binary { float apply(float left, float right); }

    private CpuNdElementwise() {}

    public static boolean matrix(Tensor input) {
        return input.rank() == 2 && input.dtype() == DType.FLOAT32 && input.coversStorage();
    }

    public static boolean sameMatrices(Tensor left, Tensor right) {
        return matrix(left) && matrix(right) && left.layout().shape().equals(right.layout().shape());
    }

    public static Tensor unary(Tensor input, Unary operation) {
        Tensor output = new Tensor(input.dtype(), input.shape());
        unaryInto(input, output, operation);
        return output;
    }

    public static void unaryInto(Tensor input, Tensor output, Unary operation) {
        Tensor.requireSameShape(input, output, "unary output");
        output.layout().requireWritable();
        input = readable(input, output);
        input.materialize();
        output.materialize();
        if (CpuDenseElementwise.supports(input, output)) {
            CpuDenseElementwise.unary(input, output, operation);
            return;
        }
        for (int index = 0; index < output.numel(); index++) {
            float value = input.memory().get(input.layout().addressAt(index));
            output.memory().set(output.layout().addressAt(index), operation.apply(value));
        }
    }

    public static Tensor binary(Tensor left, Tensor right, Binary operation) {
        TensorShape shape = ShapeTransforms.broadcast(left.layout().shape(), right.layout().shape());
        Tensor output = new Tensor(DType.promote(left.dtype(), right.dtype()), shape.dimensions());
        binaryInto(left, right, output, operation);
        return output;
    }

    public static void binaryInto(Tensor left, Tensor right, Tensor output, Binary operation) {
        TensorShape shape = ShapeTransforms.broadcast(left.layout().shape(), right.layout().shape());
        if (!shape.equals(output.layout().shape())) throw new IllegalArgumentException("Cannot expand in-place output");
        output.layout().requireWritable();
        if (DType.promote(left.dtype(), right.dtype()) != output.dtype()) {
            throw new IllegalArgumentException("In-place output cannot narrow the promoted dtype");
        }
        left = readable(left, output);
        right = readable(right, output);
        writeBinary(left, right, output, operation);
    }

    private static void writeBinary(Tensor left, Tensor right, Tensor output, Binary operation) {
        left.materialize();
        right.materialize();
        output.materialize();
        if (CpuDenseElementwise.supports(left, right, output)) {
            CpuDenseElementwise.binary(left, right, output, operation);
            return;
        }
        TensorLayout a = LayoutTransforms.expand(left.layout(), output.shape());
        TensorLayout b = LayoutTransforms.expand(right.layout(), output.shape());
        for (int index = 0; index < output.numel(); index++) {
            float value = operation.apply(left.memory().get(a.addressAt(index)),
                    right.memory().get(b.addressAt(index)));
            output.memory().set(output.layout().addressAt(index), value);
        }
    }

    private static Tensor readable(Tensor input, Tensor output) {
        if (input.memory() != output.memory()) return input;
        boolean same = input.layout().shape().equals(output.layout().shape())
                && input.storageOffset() == output.storageOffset() && Arrays.equals(input.strides(), output.strides());
        return same ? input : new Tensor(input);
    }

    public static Tensor cast(Tensor input, DType dtype) {
        Objects.requireNonNull(dtype, "dtype");
        if (input.dtype() == dtype) return input;
        input.materialize();
        Tensor result = new Tensor(dtype, input.shape());
        for (int index = 0; index < input.numel(); index++) {
            result.memory().set(index, input.memory().get(input.layout().addressAt(index)));
        }
        return result;
    }
}
