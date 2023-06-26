package org.jbackprop;

public class TanhNeuron extends Neuron{
    public TanhNeuron(int numConnections, Layer previousLayer, GlobalParams globalParams) {
        super(numConnections, previousLayer, globalParams);
    }

    @Override
    double activationFunction(double net) {
        return tanh(net);
    }

    @Override
    double dActivation(double net) {
        return tanhDerivative(net);
    }

    public double tanh(double x) {
        return Math.tanh(x);
    }

    public double tanhDerivative(double x) {
        double tanhX = Math.tanh(x);
        return 1 - Math.pow(tanhX, 2);
    }
}
