package org.jbackprop.ann.lossfunctions;

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
    public double derivative(double actualLoss){
        return actualLoss;
    }

    private static double calculateSquaredError(double actualLoss){
        return actualLoss*actualLoss;
    }
}
