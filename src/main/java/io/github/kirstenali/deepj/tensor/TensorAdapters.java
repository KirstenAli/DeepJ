package io.github.kirstenali.deepj.tensor;

/**
 * Utilities for converting between {@link Tensor} ({@code double[][]}) and
 * flat {@code float[]} arrays used by GPU runtimes.
 */
public final class TensorAdapters {

    private TensorAdapters() {}

    /** Pack a Tensor's data into a flat float32 array (row-major). */
    public static float[] packF32(Tensor t) {
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

    /** Unpack a flat float32 array into a new Tensor. */
    public static Tensor unpackF32(float[] flat, int rows, int cols) {
        if (flat.length != rows * cols) {
            throw new IllegalArgumentException(
                    "Flat buffer length " + flat.length +
                            " does not match shape " + rows + "x" + cols);
        }

        Tensor t = new Tensor(rows, cols);
        unpackF32Into(flat, t);
        return t;
    }

    /**
     * Unpack a flat float32 array into an existing tensor's data[][].
     * Used by {@link ComputeGraph} to materialize GPU results without allocating a new Tensor.
     */
    public static void unpackF32Into(float[] flat, Tensor t) {
        int i = 0;
        for (int r = 0; r < t.rows; r++) {
            double[] row = t.data[r];
            for (int c = 0; c < t.cols; c++) {
                row[c] = flat[i++];
            }
        }
    }
}

