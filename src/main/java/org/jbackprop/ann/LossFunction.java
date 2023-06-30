package org.jbackprop.ann;

public abstract class LossFunction {
    abstract double calculateLoss(double target, double actual);
    abstract double dLoss(double target, double actual);
    abstract double getSumLoss(Layer outputLayer);
    double calculateSumLoss(Layer outputLayer){
        double sumLoss =0;
        var neurons = outputLayer.getNeurons();

        for (Neuron neuron : neurons){
            sumLoss += neuron.getLoss();
        }
        return sumLoss;
    }
}
