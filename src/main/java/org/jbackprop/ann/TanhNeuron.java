package org.jbackprop.ann;

public class TanhNeuron extends Neuron{
    public TanhNeuron(Integer numConnections, Layer previousLayer, NetworkParams networkParams) {
        super(numConnections, previousLayer, networkParams);
    }

    @Override
    double activationFunction(double net) {
        return tanh(net);
    }

    @Override
    double dActivation(double net) {
        return tanhDerivative(net);
    }

    public static double tanh(double x) {
        return Math.tanh(x);
    }

    public static double tanhDerivative(double x) {
        double tanhX = Math.tanh(x);
        return 1 - Math.pow(tanhX, 2);
    }
}
