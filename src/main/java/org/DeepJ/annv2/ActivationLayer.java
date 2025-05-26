package org.DeepJ.annv2;

import org.DeepJ.annv2.activations.ActivationFunction;
import org.DeepJ.transformer.Tensor;

public class ActivationLayer implements Layer {
    private final ActivationFunction activation;
    private Tensor input;

    public ActivationLayer(ActivationFunction activation) {
        this.activation = activation;
    }

    @Override
    public Tensor forward(Tensor input) {
        this.input = input;
        return activation.forward(input);
    }

    @Override
    public Tensor backward(Tensor gradOutput, double learningRate) {
        return activation.backward(gradOutput);
    }
}