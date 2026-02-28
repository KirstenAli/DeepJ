package org.deepj.ann.layers;

import org.deepj.ann.Tensor;
import org.deepj.ann.training.Trainable;

/**
 * Differentiable module mapping Tensor -> Tensor.
 * Layers may expose parameters via Trainable (defaults to none).
 */
public interface Layer extends Trainable {

    Tensor forward(Tensor input);

    Tensor backward(Tensor gradOutput, double learningRate);
}