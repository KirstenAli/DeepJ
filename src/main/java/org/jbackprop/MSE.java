package org.jbackprop;

public class MSE extends LossFunction {
    @Override
    public double calculateLoss(double target, double actual){
        return calculateMSE(target, actual);
    }

    @Override
    public double dLoss(double target, double actual){
        return calculateMSEDerivative(target, actual);
    }

    @Override
    double getSumLoss(Layer outputLayer) {
        return 0;
    }

    private static double calculateMSE(double target, double actual){
        return Math.pow((target - actual), 2);
    }

    private static double calculateMSEDerivative(double target, double actual){
        return 2 * (target - actual);
    }
}
