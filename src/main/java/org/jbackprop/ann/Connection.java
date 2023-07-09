package org.jbackprop.ann;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.Setter;

import java.io.Serializable;

@Setter
@Getter
public class Connection implements Serializable {
    private Neuron outputNeuron;
    private Neuron inputNeuron;
    private double input;
    @JsonProperty
    private double weight;
    private double product;
    private double learningRate;
    private double momentum;
    private double prevUpdate;
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
