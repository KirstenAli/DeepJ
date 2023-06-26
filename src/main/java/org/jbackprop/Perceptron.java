package org.jbackprop;

public class Perceptron extends Neuron{
    public Perceptron(int numConnections, Layer previousLayer, GlobalParams globalParams) {
        super(numConnections, previousLayer, globalParams);
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

    @Override
    public double calculateLoss(double target) {
        var loss = super.calculateLoss(target);
        setDelta(loss*-1);

        return loss;
    }
}
