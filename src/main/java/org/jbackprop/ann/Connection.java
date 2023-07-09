package org.jbackprop.ann;

import com.fasterxml.jackson.annotation.JsonIgnore;

import lombok.Getter;
import lombok.Setter;

@Setter
@Getter
public class Connection {
    @JsonIgnore
    private Neuron outputNeuron;
    @JsonIgnore
    private Neuron inputNeuron;
    private double input;
    private double weight;
    private double product;
    private double learningRate;
    private double momentum;
    private double prevUpdate;
    @JsonIgnore
    private NetworkBuilder networkBuilder;

    public Connection(NetworkBuilder networkBuilder) {
        weight = Math.random() - 0.5;
        input = 1;
        learningRate = networkBuilder.getLearningRate();
        momentum = networkBuilder.getMomentum();
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
        var update = learningRate * delta * input;
        update+= momentum*prevUpdate; // add momentum to update
        weight += update;
        prevUpdate = update;
    }

}
