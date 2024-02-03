package org.DeepJ.ann.activationfunctions;
public class Tanh implements ActivationFunction {

    @Override
    public double applyActivation(double net) {
        return tanh(net);
    }

    @Override
    public double derivative(double net, double activation) {
        return tanhDerivative(net, activation);
    }

    public static double tanh(double x) {
        return Math.tanh(x);
    }

    public static double tanhDerivative(double x, double activation) {

        return 1 - Math.pow(activation, 2);
    }
}
