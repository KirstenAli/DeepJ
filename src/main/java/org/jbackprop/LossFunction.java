package org.jbackprop;

public abstract class LossFunction {
    abstract double calculateLoss(double target, double actual);
    abstract double dLoss(double target, double actual);
    abstract double getSumError(Layer outputLayer);
    double calculateSumError(Layer outputLayer) {
        double sumError =0;
        var neurons = outputLayer.getNeurons();

        for (Neuron neuron : neurons) {
            sumError += neuron.getLoss();
        }

        return sumError;
    }
}
