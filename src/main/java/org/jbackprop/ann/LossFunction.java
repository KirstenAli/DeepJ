package org.jbackprop.ann;

public abstract class LossFunction {
    abstract double calculateLoss(double loss);
    abstract double dLoss(double loss);
    abstract double calculateSumLoss(Layer outputLayer);
    double calculateActualSumLoss(Layer outputLayer){
        double sumLoss =0;
        var neurons = outputLayer.getNeurons();

        for (Neuron neuron : neurons){
            sumLoss += neuron.calculateLoss();
        }
        return sumLoss;
    }
}
