package io.github.kirstenali.deepj.tensor;

/** Contiguous compact rotary values retain their storage dtype without per-element layout division. */
final class CpuCompactRotary {
    private final short[] input, output;
    private final float[] cosine, sine;
    private final DType dtype;
    private final int offset, cosineOffset, sineOffset, width, length, step, distance;
    private final float direction;

    private CpuCompactRotary(Tensor x, Tensor c, Tensor s, Tensor y, int length,
                              boolean inverse, boolean halfSplit) {
        input = x.memory().shorts();
        output = y.memory().shorts();
        cosine = c.memory().floats();
        sine = s.memory().floats();
        dtype = x.dtype();
        offset = x.storageOffset();
        cosineOffset = c.storageOffset();
        sineOffset = s.storageOffset();
        width = x.matrixCols();
        this.length = length;
        step = halfSplit ? 1 : 2;
        distance = halfSplit ? width / 2 : 1;
        direction = inverse ? -1 : 1;
    }

    static boolean apply(Tensor input, Tensor cosine, Tensor sine, Tensor output, int length,
                           boolean inverse, boolean halfSplit) {
        if (input.dtype() == DType.FLOAT32 || !input.isContiguous()) return false;
        if (!denseTable(cosine) || !denseTable(sine)) return false;
        var rotation = new CpuCompactRotary(input, cosine, sine, output, length, inverse, halfSplit);
        if (input.numel() >= 16384) CpuRanges.balance(input.matrixRows(), rotation::row);
        else for (int row = 0; row < input.matrixRows(); row++) { rotation.row(row); }
        return true;
    }

    private static boolean denseTable(Tensor table) {
        return table.dtype() == DType.FLOAT32 && table.isContiguous();
    }

    private void row(int row) {
        int base = row * width, table = row % length * (width / 2);
        for (int pair = 0; pair < width / 2; pair++) { rotate(base + pair * step, table + pair); }
    }

    private void rotate(int first, int table) {
        float x = dtype.decode16(input[offset + first]), y = dtype.decode16(input[offset + first + distance]);
        float c = cosine[cosineOffset + table], s = sine[sineOffset + table] * direction;
        output[first] = dtype.encode16(x * c - y * s);
        output[first + distance] = dtype.encode16(x * s + y * c);
    }
}
