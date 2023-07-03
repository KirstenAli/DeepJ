package org.jbackprop.ann.lossfunctions;

import org.jbackprop.ann.OutputLayer;

public interface LossFunction {
    double calculateLoss(double loss);
    double derivative(double loss);
    double calculateSumLoss(OutputLayer outputLayer);
}
