package io.github.kirstenali.deepj.loss;

import io.github.kirstenali.deepj.tensor.Tensor;

/**
 * Mean-Squared-Error loss.
 *
 * <p><b>Reduction convention:</b> the loss is averaged over <em>every element</em>
 * ({@code rows × cols}), not just over rows. Consequently the gradient is
 * {@code 2·(predicted − actual) / (rows·cols)}. This keeps the loss scale independent
 * of the feature dimension. If a sum reduction or a per-row mean is required, scale the
 * result accordingly.
 */
public class MSELoss implements LossFunction {
    @Override
    public float loss(Tensor predicted, Tensor actual) {
        Tensor diff = predicted.subtract(actual);
        Tensor sq = diff.multiply(diff);
        float sse = sq.sum();
        return sse / (predicted.rows * predicted.cols);
    }

    @Override
    public Tensor gradient(Tensor predicted, Tensor actual) {
        Tensor grad = predicted.subtract(actual);
        grad.multiplyScalarInPlace(2.0f / (predicted.rows * predicted.cols));
        return grad;
    }
}