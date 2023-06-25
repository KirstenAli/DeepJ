package org.jbackprop;

public class SigmoidNeuron extends Neuron{

    public SigmoidNeuron(int numConnections, Layer previousLayer) {
        super(numConnections, previousLayer);
    }

    @Override
    Double activationFunction(double net) {
        return sigmoid(net);
    }

    @Override
    Double dActivation(double net) {
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
