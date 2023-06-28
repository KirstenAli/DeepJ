package org.jbackprop;

import lombok.Getter;
import lombok.Setter;

@Getter @Setter
public class Connection {
    private Neuron outputNeuron;
    private Neuron inputNeuron;
    private double input;
    private double weight;
    private double product;
    private Double learningRate;

    private GlobalParams globalParams;

    public Connection(GlobalParams globalParams) {
        weight = Math.random();
        input = 1;
        learningRate = globalParams.getLearningRate();
    }

    public double calculateProduct(){
        product = input*weight;
        return product;
    }

    public double calculateWeightedDelta(){
        var delta = inputNeuron.getDelta();

        return delta*weight;
    }

    public void adjustWeight(){
        var delta = inputNeuron.getDelta();
        weight+=learningRate*delta*input;
    }
}
