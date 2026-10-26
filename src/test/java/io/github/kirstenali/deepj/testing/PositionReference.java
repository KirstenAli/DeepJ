package io.github.kirstenali.deepj.testing;

import io.github.kirstenali.deepj.tensor.CompactFloatReference;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;

/** Double-precision independent equations,followed by independent compact-format rounding. */
public final class PositionReference {
    private PositionReference() {}

    public static Tensor input(DType dtype) {
        float[] values = new float[32];
        for (int index = 0; index < values.length; index++) { values[index] = (index % 13 - 6) * 0.125f; }
        return Tensor.from(values, dtype, 4, 8).permute(1, 0);
    }

    public static Tensor table(DType dtype, boolean sine) {
        float[] values = new float[8];
        for (int index = 0; index < values.length; index++) {
            values[index] = (float) (sine ? Math.sin(index * 0.17) : Math.cos(index * 0.17));
        }
        return Tensor.from(values, dtype, 2, 4).permute(1, 0);
    }

    public static float[] mask(Tensor input, int length) {
        float[] result = input.toFloatArray();
        for (int index = 0; index < result.length; index++) {
            if (index % length > index / length % length) result[index] = rounded(-1e9f, input.dtype());
        }
        return result;
    }

    public static float[] softmax(Tensor input, int length, float scale) {
        float[] values = input.toFloatArray(), result = new float[values.length];
        for (int row = 0; row < input.matrixRows(); row++) {
            softmaxRow(values, result, row, length, scale, input.dtype());
        }
        return result;
    }

    private static void softmaxRow(float[] input, float[] output, int row, int width, float scale, DType dtype) {
        double maximum = Double.NEGATIVE_INFINITY, sum = 0;
        int base = row * width, count = row % width + 1;
        for (int column = 0; column < count; column++) {
            maximum = Math.max(maximum, input[base + column] * (double) scale);
        }
        for (int column = 0; column < count; column++) {
            sum += Math.exp(input[base + column] * (double) scale - maximum);
        }
        for (int column = 0; column < count; column++) {
            double probability = Math.exp(input[base + column] * (double) scale - maximum) / sum;
            output[base + column] = rounded((float) probability, dtype);
        }
    }

    public static float[] rotary(Tensor input, Tensor cosine, Tensor sine, int length, boolean inverse) {
        float[] x = input.toFloatArray(), c = cosine.toFloatArray(), s = sine.toFloatArray();
        float[] result = new float[x.length];
        for (int index = 0; index < x.length; index += 2) {
            int table = index / input.matrixCols() % length * (input.matrixCols() / 2) + index % input.matrixCols() / 2;
            double sin = s[table] * (inverse ? -1.0 : 1.0);
            result[index] = rounded((float) (x[index] * (double) c[table] - x[index + 1] * sin), input.dtype());
            result[index + 1] = rounded((float) (x[index] * sin + x[index + 1] * (double) c[table]), input.dtype());
        }
        return result;
    }

    public static float rounded(float value, DType dtype) {
        return switch (dtype) {
            case FLOAT32 -> value;
            case FLOAT16 -> CompactFloatReference.half(Short.toUnsignedInt(Float.floatToFloat16(value)));
            case BFLOAT16 -> Float.intBitsToFloat(Short.toUnsignedInt(CompactFloatReference.bfloat(value)) << 16);
        };
    }
}
