package io.github.kirstenali.deepj.layers;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.training.Trainable;
import io.github.kirstenali.deepj.inference.InferenceContext;

public interface Layer extends Trainable {

    Tensor forward(Tensor input);

    default Tensor forward(Tensor input, InferenceContext context) { return forward(input); }

    default boolean supportsIncrementalInference() { return false; }
}
