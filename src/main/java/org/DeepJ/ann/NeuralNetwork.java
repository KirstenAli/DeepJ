package org.DeepJ.ann;

import org.DeepJ.ann.layers.Layer;
import org.DeepJ.ann.loss.LossFunction;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    private final List<Layer> layers = new ArrayList<>();

    private final Tensor input;
    private final Tensor target;
    private final LossFunction lossFunction;
    private final int epochs;
    private final double targetLoss;
    private final double learningRate;
    private final boolean logLoss;

    public NeuralNetwork(
            Tensor input,
            Tensor target,
            LossFunction lossFunction,
            int epochs, double targetLoss,
            double learningRate,
            boolean logLoss
    ) {
        this.input = input;
        this.target = target;
        this.lossFunction = lossFunction;
        this.epochs = epochs;
        this.targetLoss = targetLoss;
        this.learningRate = learningRate;
        this.logLoss = logLoss;
    }

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

    public void step() {
        for (Layer layer : layers) {
            layer.step();
        }
    }

    public void train() {
        for (int epoch = 0; epoch < epochs; epoch++) {
            Tensor output = forward(input);
            double loss = lossFunction.loss(output, target);
            Tensor grad = lossFunction.gradient(output, target);
            backward(grad, learningRate);
            step();

            if (logLoss) {
                System.out.printf("Epoch %d, Loss: %.6f%n", epoch, loss);
            }

            if (loss <= targetLoss) {
                if (logLoss) {
                    System.out.printf("target loss %.6f reached at epoch %d%n", targetLoss, epoch);
                }
                break;
            }
        }
    }
}
