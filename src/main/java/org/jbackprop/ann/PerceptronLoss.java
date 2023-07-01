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
    double calculateSumLoss(Layer outputLayer) {
        return calculateActualSumLoss(outputLayer);
    }
}
