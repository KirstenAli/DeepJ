package io.github.kirstenali.deepj.models;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.training.Trainable;

public interface CausalLM extends Trainable {

    Tensor forward(int[] inputIds);

    void backward(Tensor dLogits);

    float gradClipNorm();
}
