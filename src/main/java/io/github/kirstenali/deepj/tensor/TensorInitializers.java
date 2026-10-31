package io.github.kirstenali.deepj.tensor;

import java.util.Objects;
import java.util.Random;

/** Initializes storage directly, without a full FLOAT32 staging array for compact tensors. */
final class TensorInitializers {
    private TensorInitializers() {}

    static Tensor ones(DType dtype, int... shape) {
        Tensor tensor = new Tensor(dtype, shape);
        for (int index = 0; index < tensor.numel(); index++) {
            tensor.memory().set(index, 1.0f);
        }
        return tensor;
    }

    static Tensor random(DType dtype, Random random, int... shape) {
        Objects.requireNonNull(random, "random");
        Tensor tensor = new Tensor(dtype, shape);
        for (int index = 0; index < tensor.numel(); index++) {
            tensor.memory().set(index, (float) (random.nextGaussian() * 0.1));
        }
        return tensor;
    }
}
