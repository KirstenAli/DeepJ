package org.jbackprop;

public class TanhNeuron extends Neuron{
    public TanhNeuron(int numConnections, Layer previousLayer, GlobalParams globalParams) {
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
