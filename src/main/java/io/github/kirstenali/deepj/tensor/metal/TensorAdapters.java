package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;

final class TensorAdapters {

    private TensorAdapters() {}

    static float[] packF32(Tensor t) {
        float[] out = new float[t.rows * t.cols];
        int i = 0;
        for (int r = 0; r < t.rows; r++) {
            double[] row = t.data[r];
            for (int c = 0; c < t.cols; c++) {
                out[i++] = (float) row[c];
            }
        }
        return out;
    }

    static Tensor unpackF32(float[] flat, int rows, int cols) {
        if (flat.length != rows * cols) {
            throw new IllegalArgumentException(
                    "Flat buffer length " + flat.length +
                            " does not match shape " + rows + "x" + cols);
        }

        Tensor t = new Tensor(rows, cols);
        int i = 0;
        for (int r = 0; r < rows; r++) {
            double[] row = t.data[r];
            for (int c = 0; c < cols; c++) {
                row[c] = flat[i++];
            }
        }
        return t;
    }
}