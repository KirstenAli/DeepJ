package org.jbackprop.ann;

import lombok.Getter;
import lombok.Setter;

@Setter
@Getter
public class Connection {
    private Neuron outputNeuron;
    private Neuron inputNeuron;
    private double input;
    private double weight;
    private double product;
    private double learningRate;

    private NetworkBuilder networkBuilder;

    public Connection(NetworkBuilder networkBuilder) {
        weight = Math.random() - 0.5;
        input = 1;
        learningRate = networkBuilder.getLearningRate();
    }

    public double calculateProduct() {
        product = input * weight;
        return product;
    }

    public double calculateWeightedDelta() {
        var delta = inputNeuron.getDelta();

        return delta * weight;
    }

    public void adjustWeight() {
        var delta = inputNeuron.getDelta();
        weight += learningRate * delta * input;
    }

}
