package io.github.kirstenali.deepj.tensor;

/** Utilities for converting between {@link Tensor} and flat GPU float arrays. */
public final class TensorAdapters {

    private TensorAdapters() {}

    /** Pack a Tensor's data into a flat float32 array (row-major). */
    public static float[] packF32(Tensor t) {
        t.materialize();
        return java.util.Arrays.copyOf(t.data, t.data.length);
    }

    /** Build a [n x 1] tensor from integer ids (stored as float values). */
    public static Tensor fromIntColumn(int[] values) {
        Tensor t = new Tensor(values.length, 1);
        for (int i = 0; i < values.length; i++) {
            t.data[i] = values[i];
        }
        return t;
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
     * Unpack a flat float32 array into an existing tensor's data[].
     * Used by {@link ComputeGraph} to materialize GPU results without allocating a new Tensor.
     */
    public static void unpackF32Into(float[] flat, Tensor t) {
        if (flat.length != t.data.length) {
            throw new IllegalArgumentException(
                    "Flat buffer length " + flat.length +
                            " does not match tensor size " + t.data.length);
        }
        System.arraycopy(flat, 0, t.data, 0, flat.length);
    }
}
