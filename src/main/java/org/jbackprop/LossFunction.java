package org.jbackprop;

import java.util.List;

public interface LossFunction {
    double calculateLoss(double target, double actual);
    double dLoss(double target, double actual);
    double calculateSumError(Layer outputlayer, List<Double> target);
}
