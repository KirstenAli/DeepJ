package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.util.Objects;
import java.util.Random;

/** Inverted dropout. Only random mask generation uses the host; tensor arithmetic uses the backend. */
public final class Dropout extends Module {
    private final float probability;
    private final Random random;

    public Dropout(float probability, long seed) { this(probability, new Random(seed)); }

    public Dropout(float probability, Random random) {
        NeuralChecks.probability(probability, "dropout probability");
        this.probability = probability;
        this.random = Objects.requireNonNull(random, "random");
    }

    @Override public Tensor forward(Tensor input) {
        if (!isTraining() || probability == 0) return input;
        float[] values = new float[input.numel()];
        float scale = 1.0f / (1 - probability);
        for (int index = 0; index < values.length; index++) {
            values[index] = random.nextFloat() < probability ? 0 : scale;
        }
        try (var scope = TensorScope.open()) {
            return scope.escape(input.multiply(Tensor.from(values, input.dtype(), input.shape())));
        }
    }
}
