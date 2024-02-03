package org.DeepJ.ann.activationfunctions;

public class Sigmoid implements ActivationFunction {

    @Override
    public double applyActivation(double net) {
        return sigmoid(net);
    }

    @Override
    public double derivative(double net, double activation) {
        return sigmoidDerivative(net, activation);
    }

    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double sigmoidDerivative(double x, double activation) {
        return activation * (1 - activation);
    }
}
