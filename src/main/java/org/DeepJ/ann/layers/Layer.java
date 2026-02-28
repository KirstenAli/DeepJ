package org.DeepJ.ann.layers;

import org.DeepJ.ann.Tensor;
import org.DeepJ.ann.training.Trainable;

/**
 * Differentiable module mapping Tensor -> Tensor.
 * Layers may expose parameters via Trainable (defaults to none).
 */
public interface Layer extends Trainable {

    Tensor forward(Tensor input);

    Tensor backward(Tensor gradOutput, double learningRate);
}