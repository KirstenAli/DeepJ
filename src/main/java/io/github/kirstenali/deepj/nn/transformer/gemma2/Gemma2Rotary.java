package io.github.kirstenali.deepj.nn.transformer.gemma2;

import io.github.kirstenali.deepj.layers.transformer.qwen3.Qwen3Rotary;
import io.github.kirstenali.deepj.layers.transformer.qwen3.Qwen3RotaryTables;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.stream.IntStream;

/** Gemma rounds rotary products in the input dtype; FP32 retains the existing implementation. */
final class Gemma2Rotary {
    private final Qwen3Rotary fullPrecision;
    private final int width;
    private final double theta;
    private final int[] partners;

    Gemma2Rotary(int width, double theta) {
        fullPrecision = new Qwen3Rotary(width, theta);
        this.width = width;
        this.theta = theta;
        partners = IntStream.range(0, width).map(i -> (i + width / 2) % width).toArray();
    }

    Tensor apply(Tensor input, int length, int offset) {
        if (input.dtype() != DType.BFLOAT16) return fullPrecision.apply(input, length, offset);
        if (input.size(-1) != width || length < 1 || offset < 0) {
            throw new IllegalArgumentException("Invalid Gemma rotary input or positions");
        }
        Math.addExact(offset, length);
        Tensor[] tables = Qwen3RotaryTables.get("gemma2-bfloat16", width, theta, length, offset,
                () -> tables(length, offset));
        Tensor rotated = input.indexSelect(-1, partners).multiply(tables[2]);
        return input.multiply(tables[0]).add(rotated.multiply(tables[1]));
    }

    private Tensor[] tables(int length, int offset) {
        float[] cosine = new float[Math.multiplyExact(length, width)], sine = new float[cosine.length];
        for (int time = 0; time < length; time++) {
            for (int channel = 0; channel < width; channel++) {
                float frequency = 1.0f / (float) Math.pow(theta, 2.0 * (channel % (width / 2)) / width);
                float angle = (time + offset) * frequency;
                cosine[time * width + channel] = (float) Math.cos(angle);
                sine[time * width + channel] = (float) Math.sin(angle);
            }
        }
        return new Tensor[]{Tensor.from(cosine, DType.BFLOAT16, 1, 1, length, width),
                Tensor.from(sine, DType.BFLOAT16, 1, 1, length, width), signs()};
    }

    private Tensor signs() {
        float[] values = new float[width];
        for (int index = 0; index < width; index++) { values[index] = index < width / 2 ? -1 : 1; }
        return Tensor.from(values, DType.BFLOAT16, width);
    }
}
