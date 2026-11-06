package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.Arrays;
import java.util.Objects;

/** A DeepJ floating tensor or an exact ONNX integer/boolean tensor. */
public final class OnnxValue {
    private final Tensor tensor;
    private final long[] integers;
    private final int[] shape;
    private final int type;

    private OnnxValue(Tensor tensor, long[] integers, int[] shape, int type) {
        this.tensor = tensor;
        this.integers = integers;
        this.shape = shape.clone();
        this.type = type;
    }

    public static OnnxValue of(Tensor tensor) {
        Objects.requireNonNull(tensor, "tensor");
        int type =
                switch (tensor.dtype()) {
                    case FLOAT32 -> 1;
                    case FLOAT16 -> 10;
                    case BFLOAT16 -> 16;
                };
        return new OnnxValue(tensor, null, tensor.shape(), type);
    }

    public static OnnxValue integers(long[] values, int... shape) {
        return host(values, shape, 7);
    }

    public static OnnxValue booleans(boolean[] values, int... shape) {
        long[] data = new long[values.length];
        for (int i = 0; i < values.length; i++) {
            data[i] = values[i] ? 1 : 0;
        }
        return host(data, shape, 9);
    }

    /** Exact integer/boolean input using an ONNX TensorProto element-type number. */
    public static OnnxValue integerTensor(int elementType, long[] values, int... shape) {
        return host(values, shape, elementType);
    }

    static OnnxValue host(long[] data, int[] shape, int type) {
        IntegerValues.validate(data, type);
        if (elements(shape) != data.length)
            throw new IllegalArgumentException("ONNX value shape differs from data");
        return new OnnxValue(null, data.clone(), shape, type);
    }

    public boolean isFloating() {
        return tensor != null;
    }

    public int elementType() {
        return type;
    }

    public int[] shape() {
        return shape.clone();
    }

    public int size() {
        return tensor == null ? integers.length : tensor.numel();
    }

    public Tensor tensor() {
        if (tensor == null)
            throw new IllegalArgumentException("Expected floating ONNX tensor, got type " + type);
        return tensor;
    }

    public long[] longs() {
        return data().clone();
    }

    long[] data() {
        if (integers == null)
            throw new IllegalArgumentException("Expected integer or boolean ONNX tensor");
        return integers;
    }

    int[] ints() {
        return Arrays.stream(data()).mapToInt(Math::toIntExact).toArray();
    }

    long scalar() {
        if (size() != 1) throw new IllegalArgumentException("Expected a scalar ONNX value");
        return data()[0];
    }

    static int elements(int[] shape) {
        boolean empty = false;
        for (int dimension : shape) {
            if (dimension < 0) throw new IllegalArgumentException("Negative ONNX dimension");
            empty |= dimension == 0;
        }
        if (empty) return 0;
        int total = 1;
        for (int dimension : shape) {
            if (dimension < 0) throw new IllegalArgumentException("Negative ONNX dimension");
            total = Math.multiplyExact(total, dimension);
        }
        return total;
    }
}
