package org.jbackprop;

public class ReluNeuron extends Neuron{
    public ReluNeuron(int numConnections, Layer previousLayer, GlobalParams globalParams) {
        super(numConnections, previousLayer, globalParams);
    }

    @Override
    Double activationFunction(double net) {
        return null;
    }

    @Override
    Double dActivation(double net) {
        return null;
    }
}
