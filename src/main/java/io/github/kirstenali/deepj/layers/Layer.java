package io.github.kirstenali.deepj.layers;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.training.Trainable;

/**
 * Differentiable module mapping Tensor -> Tensor.
 * Layers may expose parameters via Trainable (defaults to none).
 */
public interface Layer extends Trainable {

    Tensor forward(Tensor input);

    Tensor backward(Tensor gradOutput);
}