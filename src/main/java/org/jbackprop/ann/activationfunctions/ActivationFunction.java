package org.jbackprop.ann.activationfunctions;

public interface ActivationFunction {
    double applyActivation(double net);

    double derivative(double net, double activation);
}
