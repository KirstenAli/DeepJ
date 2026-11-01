package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.tensor.Tensor;

final class NeuralChecks {
    private NeuralChecks() {}

    static int positive(int value, String name) {
        if (value < 1) throw new IllegalArgumentException(name + " must be positive");
        return value;
    }

    static void probability(float value, String name) {
        if (!Float.isFinite(value) || value < 0 || value >= 1) {
            throw new IllegalArgumentException(name + " must be finite and in [0, 1)");
        }
    }

    static void rank(Tensor input, int rank) {
        if (input == null || input.rank() != rank || input.numel() == 0) {
            throw new IllegalArgumentException("Expected a nonempty rank-" + rank + " tensor");
        }
    }

    static void width(Tensor input, int width) {
        if (input == null || input.rank() == 0 || input.numel() == 0 || input.size(-1) != width) {
            throw new IllegalArgumentException("Expected nonempty input with final width " + width);
        }
    }
}
