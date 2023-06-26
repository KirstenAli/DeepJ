package org.jbackprop;

public class ReluNeuron extends Neuron{
    public ReluNeuron(int numConnections, Layer previousLayer, GlobalParams globalParams) {
        super(numConnections, previousLayer, globalParams);
    }

    @Override
    double activationFunction(double net) {
        return relu(net);
    }

    @Override
    double dActivation(double net) {
        return reluDerivative(net);
    }

    public double relu(double x) {
        return Math.max(0, x);
    }

    public double reluDerivative(double x) {
        if (x <= 0) {
            return 0;
        } else {
            return 1;
        }
    }
}
