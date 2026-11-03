package io.github.kirstenali.deepj.layers.transformer.qwen3;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.autograd.LayoutOps;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.stream.IntStream;

/** Changes basis around DeepJ's existing rotary operation to implement Qwen's half-split convention. */
public final class Qwen3Rotary {
    private final int width;
    private final double theta;
    private final int[] interleaved, original;

    public Qwen3Rotary(int headDim, double theta) {
        if (headDim < 2 || headDim % 2 != 0 || !(theta > 0) || !Double.isFinite(theta)) {
            throw new IllegalArgumentException("Invalid rotary geometry");
        }
        width = headDim;
        this.theta = theta;
        interleaved = IntStream.range(0, width).map(i -> i / 2 + i % 2 * (width / 2)).toArray();
        original = IntStream.range(0, width).map(i -> i % (width / 2) * 2 + i / (width / 2)).toArray();
    }

    public Tensor apply(Tensor input, int length, int offset) {
        if (input.size(-1) != width || length < 1 || offset < 0) {
            throw new IllegalArgumentException("Invalid rotary input or positions");
        }
        Math.addExact(offset, length);
        Tensor[] tables = tables(length, offset);
        Tensor paired = input.indexSelect(-1, interleaved);
        return LayoutOps.rotary(paired, tables[0], tables[1], length, false).indexSelect(-1, original);
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
            float frequency = (float) (1 / Math.pow(theta, 2.0 * dimension / width));
            float angle = position * frequency;
            int index = time * (width / 2) + dimension;
            cosine[index] = (float) Math.cos(angle);
            sine[index] = (float) Math.sin(angle);
        }
    }
}
