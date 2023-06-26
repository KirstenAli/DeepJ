package org.jbackprop;

import lombok.Getter;
import lombok.Setter;

@Getter @Setter
public class Connection {
    private Neuron outputNeuron;
    private double input;
    private double weight;
    private double product;

    private GlobalParams globalParams;

    public Connection(GlobalParams globalParams) {
        weight = Math.random();
        this.globalParams = globalParams;
    }

    public double calculateProduct(){
        product = input*weight;
        return product;
    }

    public double calculateWeightedDelta(){
        var delta = outputNeuron.getDelta();

        return delta*weight;
    }

    public void adjustWeight(){
        var learningRate = globalParams.getLearningRate();
        var delta = outputNeuron.getDelta();

        weight-=learningRate*delta*input;
    }
}
