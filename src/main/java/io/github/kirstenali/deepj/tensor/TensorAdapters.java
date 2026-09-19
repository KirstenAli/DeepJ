package io.github.kirstenali.deepj.tensor;

public final class TensorAdapters {

    private TensorAdapters() {}

    public static float[] packF32(Tensor t) {
        t.materialize();
        return java.util.Arrays.copyOf(t.data, t.data.length);
    }

    public static Tensor fromIntColumn(int[] values) {
        Tensor t = new Tensor(values.length, 1);
        for (int i = 0; i < values.length; i++) {
            t.data[i] = values[i];
        }
        return t;
    }

    public static Tensor unpackF32(float[] flat, int rows, int cols) {
        Tensor t = new Tensor(rows, cols);
        if (flat.length != t.data.length) {
            throw new IllegalArgumentException(
                    "Flat buffer length " + flat.length +
                            " does not match shape " + rows + "x" + cols);
        }
        unpackF32Into(flat, t);
        return t;
    }

    public static void unpackF32Into(float[] flat, Tensor t) {
        if (flat.length != t.data.length) {
            throw new IllegalArgumentException(
                    "Flat buffer length " + flat.length +
                            " does not match tensor size " + t.data.length);
        }
        System.arraycopy(flat, 0, t.data, 0, flat.length);
    }
}
