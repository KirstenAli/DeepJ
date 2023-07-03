package org.jbackprop.ann.activationfunctions;

public class Relu implements ActivationFunction {
    @Override
    public double applyActivation(double net) {
        return relu(net);
    }

    @Override
    public double derivative(double net, double activation) {
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
