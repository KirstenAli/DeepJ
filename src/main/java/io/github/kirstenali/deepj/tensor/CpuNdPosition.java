package io.github.kirstenali.deepj.tensor;

/** Typed position operations accumulate in FLOAT32 and round only the final stored activation. */
public final class CpuNdPosition {
    private CpuNdPosition() {}

    public static Tensor mask(Tensor input, int length) {
        MatrixShapeChecks.requireCausal(input, length);
        input.materialize();
        Tensor output = new Tensor(input.dtype(), input.shape());
        for (int index = 0; index < output.numel(); index++) {
            float result = index % length > index / length % length ? -1e9f : value(input, index);
            output.memory().set(index, result);
        }
        return output;
    }

    public static Tensor softmax(Tensor input, int length, float scale) {
        MatrixShapeChecks.requireCausal(input, length);
        requireScale(scale);
        input.materialize();
        Tensor output = new Tensor(input.dtype(), input.shape());
        for (int row = 0; row < input.matrixRows(); row++) { softmaxRow(input, output, row, length, scale); }
        return output;
    }

    public static void requireScale(float scale) {
        if (!Float.isFinite(scale)) throw new IllegalArgumentException("Attention scale must be finite");
    }

    private static void softmaxRow(Tensor input, Tensor output, int row, int width, float scale) {
        int count = row % width + 1, base = row * width;
        float maximum = Float.NEGATIVE_INFINITY, total = 0;
        for (int column = 0; column < count; column++) {
            maximum = Math.max(maximum, value(input, base + column) * scale);
        }
        for (int column = 0; column < count; column++) {
            total += (float) Math.exp(value(input, base + column) * scale - maximum);
        }
        for (int column = 0; column < count; column++) {
            output.memory().set(base + column, (float) Math.exp(value(input, base + column) * scale - maximum) / total);
        }
    }

    public static Tensor rotary(Tensor input, Tensor cosine, Tensor sine, int length, boolean inverse) {
        MatrixShapeChecks.requireRotary(input, cosine, sine, length);
        input.materialize();
        cosine.materialize();
        sine.materialize();
        Tensor output = new Tensor(input.dtype(), input.shape());
        for (int index = 0; index < input.numel(); index += 2) {
            rotatePair(input, cosine, sine, output, index, length, inverse);
        }
        return output;
    }

    private static void rotatePair(Tensor input, Tensor cosine, Tensor sine, Tensor output,
                                    int index, int length, boolean inverse) {
        int width = input.matrixCols(), table = index / width % length * (width / 2) + index % width / 2;
        float x = value(input, index), y = value(input, index + 1);
        float c = value(cosine, table), s = value(sine, table) * (inverse ? -1 : 1);
        output.memory().set(index, x * c - y * s);
        output.memory().set(index + 1, x * s + y * c);
    }

    private static float value(Tensor input, int index) {
        return input.memory().get(input.layout().addressAt(index));
    }
}
