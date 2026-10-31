package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;

/** Packed FLOAT32 products cache validated storage and strides outside the arithmetic loops. */
final class CpuPackedMatmul {
    private CpuPackedMatmul() {}

    static Tensor apply(Tensor left, Tensor right, int batches, PackedMatmulShape shape) {
        left.materialize();
        right.materialize();
        float[] a = left.rawData(), b = right.rawData();
        Tensor output = new Tensor(batches * shape.rows(), shape.cols());
        float[] result = output.rawData();
        DeepJExecutor.forRange(0, batches * shape.rows(), index -> multiplyRow(a, b, result, shape, index));
        return output;
    }

    private static void multiplyRow(float[] left, float[] right, float[] output, PackedMatmulShape shape, int index) {
        int batch = index / shape.rows(), row = index % shape.rows(), columns = shape.cols();
        int leftColumns = shape.transposeLeft() ? shape.rows() : shape.inner();
        int rightColumns = shape.transposeRight() ? shape.inner() : columns;
        int leftBase = leftBase(shape, batch, row, leftColumns);
        int rightBase = batch * shape.rightRows() * rightColumns, outputBase = index * columns;
        int leftStep = shape.transposeLeft() ? leftColumns : 1, rightStep = shape.transposeRight() ? rightColumns : 1;
        for (int k = 0; k < shape.inner(); k++) {
            float value = left[leftBase + k * leftStep];
            int start = rightBase + (shape.transposeRight() ? k : k * rightColumns);
            for (int column = 0; column < columns; column++) {
                output[outputBase + column] += value * right[start + column * rightStep];
            }
        }
    }

    private static int leftBase(PackedMatmulShape shape, int batch, int row, int columns) {
        return batch * shape.leftRows() * columns + (shape.transposeLeft() ? row : row * columns);
    }
}
