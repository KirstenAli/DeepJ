package io.github.kirstenali.deepj.tensor;

/** Dense FLOAT32 rotary positions avoid repeated layout decoding for either pairing convention. */
final class CpuDenseRotary {
    private final float[] input, cosine, sine, output;
    private final int inputOffset, cosineOffset, sineOffset, width, length, step, distance;
    private final float direction;

    private CpuDenseRotary(Tensor x, Tensor c, Tensor s, Tensor y, int length,
                           boolean inverse, boolean halfSplit) {
        input = x.memory().floats();
        cosine = c.memory().floats();
        sine = s.memory().floats();
        output = y.memory().floats();
        inputOffset = x.storageOffset();
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
        if (!dense(input) || !dense(cosine) || !dense(sine)) return false;
        var rotation = new CpuDenseRotary(input, cosine, sine, output, length, inverse, halfSplit);
        if (input.numel() >= 16384 && input.matrixRows() > 1) {
            CpuRanges.forRange(0, input.matrixRows(), rotation::row);
        } else {
            for (int row = 0; row < input.matrixRows(); row++) { rotation.row(row); }
        }
        return true;
    }

    private static boolean dense(Tensor tensor) {
        return tensor.dtype() == DType.FLOAT32 && tensor.isContiguous();
    }

    private void row(int row) {
        int base = row * width, table = row % length * (width / 2);
        for (int pair = 0; pair < width / 2; pair++) {
            rotate(base + pair * step, table + pair);
        }
    }

    private void rotate(int first, int table) {
        float x = input[inputOffset + first], y = input[inputOffset + first + distance];
        float c = cosine[cosineOffset + table], s = sine[sineOffset + table] * direction;
        output[first] = x * c - y * s;
        output[first + distance] = x * s + y * c;
    }
}
