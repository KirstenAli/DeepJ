package org.jbackprop;

public class PerceptronLoss extends LossFunction{
    @Override
    double calculateLoss(double target, double actual) {
        return target-actual;
    }

    @Override
    double dLoss(double target, double actual) {
        return Double.NaN;
    }

    @Override
    double getSumLoss(Layer outputLayer) {
        return calculateSumLoss(outputLayer);
    }
}
