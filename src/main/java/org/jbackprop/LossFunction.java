package org.jbackprop;

import java.util.List;

public abstract class LossFunction {
    abstract double calculateLoss(double target, double actual);
    abstract double dLoss(double target, double actual);
    abstract double getSumError(Layer outputLayer, List<Double> target);
    double calculateSumError(Layer outputLayer, List<Double> target) {
        double sumError =0;
        var neurons = outputLayer.getNeurons();

        for(int i=0; i<neurons.size(); i++){
            sumError+= neurons.get(i)
                    .calculateLoss(target.get(i));
        }

        return sumError;
    }
}
