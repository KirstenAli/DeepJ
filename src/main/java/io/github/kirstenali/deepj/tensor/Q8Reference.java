package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;

final class Q8Reference {
    private Q8Reference() {}

    static void expand(short[] data, int row, int width, float[] output, int offset) {
        int words = width / 32 * 17;
        for (int block = 0; block < width / 32; block++) {
            int start = row * words + block * 17;
            float scale = Float.float16ToFloat(data[start]);
            for (int pair = 0; pair < 16; pair++) {
                short bits = data[start + pair + 1];
                output[offset + block * 32 + pair * 2] = (byte) bits * scale;
                output[offset + block * 32 + pair * 2 + 1] = (byte) (bits >>> 8) * scale;
            }
        }
    }

    static void project(short[] bits, float[] input, float[] output, int rows, int width) {
        int batch = input.length / width;
        DeepJExecutor.forRange(0, rows, row -> projectRow(bits, input, output, row, rows, width, batch));
    }

    private static void projectRow(short[] bits, float[] input, float[] output, int row, int rows,
                                   int width, int batch) {
        float[] weights = new float[width];
        expand(bits, row, width, weights, 0);
        for (int sample = 0; sample < batch; sample++) {
            output[sample * rows + row] = dot(input, sample * width, weights);
        }
    }

    private static float dot(float[] input, int offset, float[] weights) {
        float sum = 0;
        for (int i = 0; i < weights.length; i++) {
            sum = Math.fma(input[offset + i], weights[i], sum);
        }
        return sum;
    }
}
