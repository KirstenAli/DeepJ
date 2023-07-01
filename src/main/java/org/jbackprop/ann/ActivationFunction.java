package org.jbackprop.ann;

public interface ActivationFunction {
    double applyActivation(double net);

    double derivative(double net, double activation);
}
