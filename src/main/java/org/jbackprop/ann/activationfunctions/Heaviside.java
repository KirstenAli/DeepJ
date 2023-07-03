package org.jbackprop.ann.activationfunctions;

public class Heaviside implements ActivationFunction {

    @Override
    public double applyActivation(double net) {
        return step(net);
    }

    @Override
    public double derivative(double net, double activation) {
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
