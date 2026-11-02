package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.Objects;

public record TensorBatch(Tensor inputs, Tensor targets) {
    public TensorBatch {
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(targets, "targets");
        if (inputs.rank() < 1 || targets.rank() < 1 || inputs.size(0) < 1 || inputs.size(0) != targets.size(0)) {
            throw new IllegalArgumentException("Inputs and targets require matching positive first-axis counts");
        }
    }

    public int samples() { return inputs.size(0); }
}
