package io.github.kirstenali.deepj.testing;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.inference.InferenceContext;
import io.github.kirstenali.deepj.tensor.Tensor;
import static org.junit.jupiter.api.Assertions.*;

/** Independent double-precision sliding attention over unmodified sequence history. */
public final class NdCacheAssertions {
    private NdCacheAssertions() {}

    public static void assertRollingBatches() {
        Object owner = new Object();
        try (var ignored = GradMode.noGrad(); var context = new InferenceContext(4, true)) {
            for (int offset = 0; offset < 13; offset += offset == 0 ? 3 : 1) {
                int length = offset == 0 ? 3 : 1;
                context.begin(offset);
                Tensor output = context.attend(owner, heads(4, offset, length, 1), heads(2, offset, length, 2),
                        heads(2, offset, length, 3), 4, 2, 0.5f);
                context.finish();
                assertArrayEquals(new int[]{2, 4, length, 2}, output.shape());
                assertArrayEquals(reference(offset, length), output.toFloatArray(), 1e-6f);
                Tensor.backend().releaseTemporaryResources();
            }
        }
    }

    private static Tensor heads(int heads, int offset, int length, int kind) {
        float[] values = new float[2 * heads * length * 2];
        for (int index = 0; index < values.length; index++) {
            int batch = index / (heads * length * 2), head = index / (length * 2) % heads;
            values[index] = value(kind, batch, head, offset + index / 2 % length, index % 2);
        }
        return Tensor.from(values, 2, heads, length, 2);
    }

    private static float[] reference(int offset, int length) {
        float[] output = new float[2 * 4 * length * 2];
        for (int index = 0; index < output.length; index++) {
            int batch = index / (4 * length * 2), head = index / (length * 2) % 4;
            output[index] = attended(batch, head, offset + index / 2 % length, index % 2);
        }
        return output;
    }

    private static float attended(int batch, int head, int position, int column) {
        double numerator = 0, denominator = 0;
        for (int token = Math.max(0, position - 3); token <= position; token++) {
            double probability = Math.exp(score(batch, head, position, token) * 0.5);
            numerator += probability * value(3, batch, head / 2, token, column);
            denominator += probability;
        }
        return (float) (numerator / denominator);
    }

    private static double score(int batch, int head, int position, int token) {
        double result = 0;
        for (int column = 0; column < 2; column++) {
            result += (double) value(1, batch, head, position, column) * value(2, batch, head / 2, token, column);
        }
        return result;
    }

    private static float value(int kind, int batch, int head, int time, int column) {
        return (float) Math.sin(kind * 0.3 + batch * 0.7 + head * 0.11 + time * 0.23 + column * 0.41);
    }
}
