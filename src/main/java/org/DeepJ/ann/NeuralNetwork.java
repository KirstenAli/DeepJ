package org.DeepJ.ann;

import org.DeepJ.ann.loss.LossFunction;
import org.DeepJ.transformer.Tensor;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    private final List<Layer> layers = new ArrayList<>();

    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    public Tensor forward(Tensor input) {
        Tensor output = input;
        for (Layer layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }

    public void backward(Tensor gradOutput, double learningRate) {
        for (int i = layers.size() - 1; i >= 0; i--) {
            gradOutput = layers.get(i).backward(gradOutput, learningRate);
        }
    }

    public void train(Tensor input, Tensor target, LossFunction lossFunction, int epochs, double lr) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            Tensor output = forward(input);
            double loss = lossFunction.loss(output, target);
            Tensor grad = lossFunction.gradient(output, target);
            backward(grad, lr);
            if (epoch % 10 == 0)
                System.out.printf("Epoch %d, Loss: %.6f%n", epoch, loss);
        }
    }
}