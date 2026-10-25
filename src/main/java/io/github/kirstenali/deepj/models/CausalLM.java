package io.github.kirstenali.deepj.models;

import java.util.Objects;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.training.Trainable;

public interface CausalLM extends Trainable {

    Tensor forward(int[] inputIds);

    default GenerationSession openGeneration() { return this::forward; }

    default GenerationSession openGeneration(ContextWindowMode mode) {
        if (Objects.requireNonNull(mode, "mode") == ContextWindowMode.ROLLING) {
            throw new UnsupportedOperationException("This model does not support rolling cached generation");
        }
        return openGeneration();
    }

    float gradClipNorm();
}
