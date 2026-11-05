package io.github.kirstenali.deepj.nn.diffusion.sana;

import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.Random;

/** Stochastic consistency-model sampling in SANA-Sprint's trigflow parameterization. */
final class SanaScm {
    static final float SIGMA = 0.5f;
    private SanaScm() {}

    static float[] schedule(int steps) {
        if (steps < 1 || steps > 4) throw new IllegalArgumentException("SANA-Sprint supports 1–4 steps");
        float[] result = new float[steps + 1];
        for (int index = 0; index < steps; index++) {
            result[index] = 1.57080f * (steps - index) / steps;
        }
        if (steps == 2) result[1] = 1.3f;
        return result;
    }

    static float time(float angle) {
        return (float) (Math.sin(angle) / (Math.cos(angle) + Math.sin(angle)));
    }

    static Tensor modelInput(Tensor latent, float time) {
        return latent.multiplyScalar(scale(time) / SIGMA);
    }

    static Tensor denoise(Tensor latent, Tensor input, Tensor prediction, float angle) {
        float time = time(angle);
        Tensor noise = input.multiplyScalar(1 - 2 * time)
                .add(prediction.multiplyScalar(1 - 2 * time + 2 * time * time))
                .multiplyScalar(SIGMA / scale(time));
        return latent.multiplyScalar((float) Math.cos(angle)).subtract(noise.multiplyScalar((float) Math.sin(angle)));
    }

    static Tensor advance(Tensor denoised, Tensor noise, float angle) {
        return denoised.multiplyScalar((float) Math.cos(angle))
                .add(noise.multiplyScalar(SIGMA * (float) Math.sin(angle)));
    }

    static Tensor noise(Random random, int... shape) {
        int length = 1;
        for (int size : shape) { length = Math.multiplyExact(length, size); }
        float[] values = new float[length];
        for (int index = 0; index < length; index++) { values[index] = (float) random.nextGaussian(); }
        return Tensor.from(values, shape);
    }

    private static float scale(float time) { return (float) Math.sqrt(time * time + (1 - time) * (1 - time)); }
}
