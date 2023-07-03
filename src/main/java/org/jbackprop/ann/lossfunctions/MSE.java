package org.jbackprop.ann.lossfunctions;

import org.jbackprop.ann.OutputLayer;

public class MSE extends LossFunction {
    @Override
    public double calculateLoss(double loss){
        return calculateMSE(loss);
    }

    @Override
    public double derivative(double loss){
        return calculateMSEDerivative(loss);
    }

    @Override
    public double calculateSumLoss(OutputLayer outputLayer) {
        return calculateActualSumLoss(outputLayer);
    }

    private static double calculateMSE(double loss){
        return loss*loss;
    }

    private static double calculateMSEDerivative(double loss){
        return 2 * loss;
    }
}
