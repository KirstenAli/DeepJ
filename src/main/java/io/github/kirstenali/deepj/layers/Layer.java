package io.github.kirstenali.deepj.layers;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.training.Trainable;

public interface Layer extends Trainable {

    Tensor forward(Tensor input);

    Tensor backward(Tensor gradOutput);
}
