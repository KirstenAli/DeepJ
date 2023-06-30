package org.jbackprop.ann;

public class SigmoidNeuron extends Neuron{
    public SigmoidNeuron(Integer numConnections,
                         Layer previousLayer,
                         NetworkParams networkParams) {
        super(numConnections, previousLayer, networkParams);
    }

    @Override
    double activationFunction(double net) {
        return sigmoid(net);
    }

    @Override
    double dActivation(double net) {
        return sigmoidDerivative(net);
    }

    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double sigmoidDerivative(double x) {
        double sigmoidX = sigmoid(x);
        return sigmoidX * (1 - sigmoidX);
    }
}
