package org.DeepJ.annoop.activationfunctions;

import java.io.Serializable;

public interface ActivationFunction extends Serializable {
    double applyActivation(double net);

    double derivative(double net, double activation);
}
