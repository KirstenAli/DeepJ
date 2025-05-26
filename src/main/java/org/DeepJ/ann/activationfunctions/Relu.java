package org.DeepJ.ann.activationfunctions;

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
        return x > 0 ? 1 : 0;
    }
}
