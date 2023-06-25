package org.jbackprop;

import lombok.Getter;
import lombok.Setter;

@Getter @Setter
public class Connection {

    private Neuron outputNeuron;
    private double input;
    private double weight;
    private double product;

    public Connection() {
        weight = Math.random();
    }

    public double calculateProduct(){
        product = input*weight;
        return product;
    }

    public double calculateWeightedDelta(){
        var delta = outputNeuron.getDelta();

        return delta*weight;
    }
}
