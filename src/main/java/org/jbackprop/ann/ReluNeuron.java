package org.jbackprop.ann;

public class ReluNeuron extends Neuron{
    public ReluNeuron(Integer numConnections, Layer previousLayer, NetworkParams networkParams) {
        super(numConnections, previousLayer, networkParams);
    }

    @Override
    double activationFunction(double net) {
        return relu(net);
    }

    @Override
    double dActivation(double net) {
        return reluDerivative(net);
    }

    public static double relu(double x) {
        return Math.max(0, x);
    }

    public static double reluDerivative(double x) {
        if (x <= 0) {
            return 0;
        } else {
            return 1;
        }
    }
}
