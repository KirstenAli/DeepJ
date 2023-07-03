package org.jbackprop.ann.lossfunctions;

import org.jbackprop.ann.OutputLayer;

public class MSE extends LossFunction {
    @Override
    public double calculateActualLoss(double target, double actual){
        return target-actual;
    }

    @Override
    public double calculateLoss(double actualLoss){
        return calculateSquaredError(actualLoss);
    }

    @Override
    public double calculateLossOfIteration(OutputLayer outputLayer) {
        return calculateSumLoss(outputLayer)/outputLayer.getLayerSize();
    }

    @Override
    public double derivative(double actualLoss){
        return squaredErrorDerivative(actualLoss);
    }

    private static double calculateSquaredError(double actualLoss){
        return actualLoss*actualLoss;
    }

    private static double squaredErrorDerivative(double actualLoss){
        return 2 * actualLoss;
    }
}
