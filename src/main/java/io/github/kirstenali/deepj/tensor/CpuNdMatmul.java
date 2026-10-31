package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;

/**
 * General strided/broadcast CPU matmul; the existing FLOAT32 matrix fast path remains separate.
 * Operands are staged once into contiguous FLOAT32 arrays so compact or strided inputs run the same
 * parallel row kernel as dense FLOAT32; accumulation is FLOAT32 and only the result is rounded.
 */
public final class CpuNdMatmul {
    private CpuNdMatmul() {}

    public static Tensor apply(Tensor left, Tensor right) {
        var shape = new NdMatmulShape(left, right);
        Tensor output = new Tensor(DType.promote(left.dtype(), right.dtype()), shape.output().dimensions());
        if (output.numel() == 0 || shape.inner() == 0) return output;
        left.materialize();
        right.materialize();
        int batches = output.numel() / (shape.rows() * shape.columns());
        float[] a = stage(left, shape.left(), batches * shape.rows() * shape.inner());
        float[] b = stage(right, shape.right(), batches * shape.inner() * shape.columns());
        float[] result = new float[output.numel()];
        DeepJExecutor.forRange(0, batches * shape.rows(), row -> row(a, b, result, row, shape));
        TensorMemory memory = output.memory();
        DeepJExecutor.forRange(0, result.length, index -> memory.set(index, result[index]));
        return output;
    }

    private static float[] stage(Tensor tensor, TensorLayout layout, int count) {
        float[] values = new float[count];
        TensorMemory memory = tensor.memory();
        if (layout.contiguous()) {
            int offset = layout.offset();
            DeepJExecutor.forRange(0, count, index -> values[index] = memory.get(offset + index));
        } else {
            DeepJExecutor.forRange(0, count, index -> values[index] = memory.get(layout.addressAt(index)));
        }
        return values;
    }

    private static void row(float[] a, float[] b, float[] out, int row, NdMatmulShape shape) {
        int inner = shape.inner(), columns = shape.columns();
        int batch = row / shape.rows();
        int aBase = row * inner, bBatch = batch * inner * columns, outBase = row * columns;
        for (int k = 0; k < inner; k++) {
            float value = a[aBase + k];
            int bBase = bBatch + k * columns;
            for (int column = 0; column < columns; column++) {
                out[outBase + column] += value * b[bBase + column];
            }
        }
    }
}
