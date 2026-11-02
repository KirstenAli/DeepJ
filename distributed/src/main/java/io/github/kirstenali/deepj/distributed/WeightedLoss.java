package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.Objects;

/** A differentiable sum of local objectives and its contribution count, before global normalization. */
public record WeightedLoss(Tensor sum, long weight) {
    public WeightedLoss {
        Objects.requireNonNull(sum, "sum");
        if (sum.numel() != 1 || weight < 0) {
            throw new IllegalArgumentException("Scalar loss and nonnegative weight required");
        }
    }

    public static WeightedLoss mean(Tensor mean, long weight) {
        return new WeightedLoss(mean.multiplyScalar(weight), weight);
    }

    public static WeightedLoss empty() { return new WeightedLoss(Tensor.from(new float[]{0}), 0); }
}
