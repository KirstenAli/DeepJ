package org.jbackprop;

import lombok.Getter;
import lombok.Setter;

@Getter @Setter
public class Connection {

    private Neuron inputNeuron;
    private Neuron outputNeuron;
    private double input;
    private double weight;
    private double product;

    public Connection() {
        weight = Math.random();
    }

    public double calculateProduct(){
        return product = input*weight;
    }
}
