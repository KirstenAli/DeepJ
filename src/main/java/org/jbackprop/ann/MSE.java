package org.jbackprop.ann;

public class MSE extends LossFunction {
    @Override
    public double calculateLoss(double loss){
        return calculateMSE(loss);
    }

    @Override
    public double dLoss(double loss){
        return calculateMSEDerivative(loss);
    }

    @Override
    double getSumLoss(Layer outputLayer) {
        return 0;
    }

    private static double calculateMSE(double loss){
        return loss*loss;
    }

    private static double calculateMSEDerivative(double loss){
        return 2 * loss;
    }
}
