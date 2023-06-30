package org.jbackprop.ann;

public class Perceptron extends Neuron{
    public Perceptron(Integer numConnections, Layer previousLayer, NetworkParams networkParams) {
        super(numConnections, previousLayer, networkParams);
    }

    @Override
    double activationFunction(double net) {
        return step(net);
    }

    @Override
    double dActivation(double net) {
        return Double.NaN;
    }

    public static int step(double x) {
        if (x >= 0) {
            return 1;
        } else {
            return 0;
        }
    }
}
