package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.autograd.GradMode;

public final class TensorAdapters {

    private TensorAdapters() {}

    public static float[] packF32(Tensor t) {
        if (t.data == null && t.getGpuTag() == null) return new float[TensorStorage.checkedSize(t.rows, t.cols)];
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
        int size = TensorStorage.checkedSize(t.rows, t.cols);
        if (flat.length != size) {
            throw new IllegalArgumentException(
                    "Flat buffer length " + flat.length +
                            " does not match tensor size " + size);
        }
        try (var ignored = GradMode.noGrad()) {
            t.materialize();
            t.gradientState().beforeMutation();
            System.arraycopy(flat, 0, t.hostStorage(), 0, flat.length);
            TensorStorage.markGpuNeedsUpload(t);
        }
    }
}
