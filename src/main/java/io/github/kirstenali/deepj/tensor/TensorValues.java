package io.github.kirstenali.deepj.tensor;

import java.util.Objects;

/** Explicit logical indexing and copying; never interprets a strided view as a flat backing array. */
final class TensorValues {
    private TensorValues() {}

    static float get(TensorStorageAccess tensor, int... indices) {
        return read(tensor, tensor.layout().address(indices));
    }

    static float get(TensorStorageAccess tensor, int row, int column) {
        return read(tensor, tensor.layout().address(row, column));
    }

    private static float read(TensorStorageAccess tensor, int address) {
        tensor.materialize();
        return tensor.memory().get(address);
    }

    static void set(TensorStorageAccess tensor, float value, int... indices) {
        write(tensor, value, tensor.layout().address(indices));
    }

    static void set(TensorStorageAccess tensor, int row, int column, float value) {
        write(tensor, value, tensor.layout().address(row, column));
    }

    private static void write(TensorStorageAccess tensor, float value, int address) {
        tensor.layout().requireWritable();
        tensor.gradientState().beforeMutation();
        tensor.materialize();
        tensor.memory().set(address, value);
        TensorStorage.markGpuNeedsUpload(tensor);
    }

    static float item(TensorStorageAccess tensor) {
        if (tensor.numel() != 1) throw new IllegalStateException("item requires a single-element tensor");
        tensor.materialize();
        return tensor.memory().get(tensor.layout().addressAt(0));
    }

    static float[] toFloatArray(TensorStorageAccess tensor) {
        tensor.materialize();
        float[] result = new float[tensor.numel()];
        for (int index = 0; index < result.length; index++) {
            result[index] = tensor.memory().get(tensor.layout().addressAt(index));
        }
        return result;
    }

    static Tensor from(float[] values, DType dtype, int... shape) {
        Objects.requireNonNull(values, "values");
        Tensor result = new Tensor(dtype, shape);
        if (values.length != result.numel()) throw new IllegalArgumentException("Values must match the tensor shape");
        for (int index = 0; index < values.length; index++) {
            result.memory().set(index, values[index]);
        }
        return result;
    }

    static Tensor vector(float[] values, DType dtype) {
        Objects.requireNonNull(values, "values");
        return from(values, dtype, values.length);
    }

    static TensorMemory copy(TensorStorageAccess source) {
        source.materialize();
        TensorLayout layout = source.layout();
        if (layout.contiguous() && layout.offset() == 0 && source.numel() == source.memory().length()) {
            return source.memory().copy();
        }
        TensorMemory result = new TensorMemory(source.dtype(), source.numel());
        int block = contiguousBlock(layout);
        for (int index = 0; index < source.numel(); index += block) {
            source.memory().copyTo(result, layout.addressAt(index), index, block);
        }
        return result;
    }

    private static int contiguousBlock(TensorLayout layout) {
        int block = 1;
        for (int axis = layout.shape().rank() - 1; axis >= 0; axis--) {
            int size = layout.shape().size(axis);
            if (size == 1) continue;
            if (size == 0 || layout.stride(axis) != block) break;
            block *= size;
        }
        return block;
    }
}
