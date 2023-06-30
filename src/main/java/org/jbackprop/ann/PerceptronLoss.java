package org.jbackprop.ann;

public class PerceptronLoss extends LossFunction{
    @Override
    double calculateLoss(double loss) {
        return loss;
    }

    @Override
    double dLoss(double loss) {
        return Double.NaN;
    }

    @Override
    double getSumLoss(Layer outputLayer) {
        return calculateSumLoss(outputLayer);
    }
}
