package org.jbackprop.ann;

public abstract class LossFunction {
    abstract double calculateLoss(double loss);
    abstract double derivative(double loss);
    abstract double calculateSumLoss(OutputLayer outputLayer);
    double calculateActualSumLoss(OutputLayer outputLayer){
        double sumLoss =0;
        var neurons = outputLayer.getNeurons();

        for (OutputNeuron neuron : neurons){
            sumLoss += neuron.calculateLoss();
        }
        return sumLoss;
    }
}
