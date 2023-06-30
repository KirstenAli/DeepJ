package org.jbackprop.ann;

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

    private NetworkParams networkParams;

    public Connection(NetworkParams networkParams) {
        weight = Math.random() -0.5;
        input = 1;
        learningRate = networkParams.getLearningRate();
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
