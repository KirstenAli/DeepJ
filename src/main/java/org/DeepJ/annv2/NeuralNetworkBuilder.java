package org.DeepJ.annv2;

import org.DeepJ.annv2.layers.Layer;
import org.DeepJ.annv2.loss.LossFunction;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetworkBuilder {
    private Tensor input;
    private Tensor target;
    private LossFunction lossFunction;
    private int epochs = 1000;
    private double learningRate = 0.01;
    private boolean logLoss = true;

    private final List<Layer> layers = new ArrayList<>();

    public NeuralNetworkBuilder input(Tensor input) {
        this.input = input;
        return this;
    }

    public NeuralNetworkBuilder target(Tensor target) {
        this.target = target;
        return this;
    }

    public NeuralNetworkBuilder loss(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
        return this;
    }

    public NeuralNetworkBuilder epochs(int epochs) {
        this.epochs = epochs;
        return this;
    }

    public NeuralNetworkBuilder learningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public NeuralNetworkBuilder logLoss(boolean logLoss) {
        this.logLoss = logLoss;
        return this;
    }

    public NeuralNetworkBuilder addLayer(Layer layer) {
        layers.add(layer);
        return this;
    }

    public NeuralNetwork build() {
        if (input == null || target == null || lossFunction == null) {
            throw new IllegalStateException("Input, target, and loss function must be set.");
        }

        NeuralNetwork net = new NeuralNetwork(input, target, lossFunction, epochs, learningRate, logLoss);
        for (Layer layer : layers) {
            net.addLayer(layer);
        }
        return net;
    }
}
