package io.github.kirstenali.deepj.layers.transformer.qwen3;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.autograd.HalfSplitRotary;
import io.github.kirstenali.deepj.tensor.Tensor;

/** Qwen rotary positions use direct half-split pairs and shared per-forward tables. */
public final class Qwen3Rotary {
    private final int width;
    private final double theta;
    private final float[] frequencies;

    public Qwen3Rotary(int headDim, double theta) {
        if (headDim < 2 || headDim % 2 != 0 || !(theta > 0) || !Double.isFinite(theta)) {
            throw new IllegalArgumentException("Invalid rotary geometry");
        }
        width = headDim;
        this.theta = theta;
        frequencies = new float[width / 2];
        for (int index = 0; index < frequencies.length; index++) {
            frequencies[index] = (float) (1 / Math.pow(theta, 2.0 * index / width));
        }
    }

    public Tensor apply(Tensor input, int length, int offset) {
        if (input.size(-1) != width || length < 1 || offset < 0) {
            throw new IllegalArgumentException("Invalid rotary input or positions");
        }
        Math.addExact(offset, length);
        Tensor[] tables = Qwen3RotaryTables.get(width, theta, length, offset, () -> tables(length, offset));
        return HalfSplitRotary.apply(input, tables[0], tables[1], length);
    }

    private Tensor[] tables(int length, int offset) {
        float[] cosine = new float[Math.multiplyExact(length, width / 2)], sine = new float[cosine.length];
        for (int time = 0; time < length; time++) {
            fill(cosine, sine, time, Math.addExact(offset, time));
        }
        try (var ignored = GradMode.noGrad()) {
            return new Tensor[]{Tensor.from(cosine, length, width / 2), Tensor.from(sine, length, width / 2)};
        }
    }

    private void fill(float[] cosine, float[] sine, int time, int position) {
        for (int dimension = 0; dimension < width / 2; dimension++) {
            float frequency = frequencies[dimension];
            float angle = position * frequency;
            int index = time * (width / 2) + dimension;
            cosine[index] = (float) Math.cos(angle);
            sine[index] = (float) Math.sin(angle);
        }
    }
}
