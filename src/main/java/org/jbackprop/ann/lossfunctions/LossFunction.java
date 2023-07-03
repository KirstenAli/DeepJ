package org.jbackprop.ann.lossfunctions;

import org.jbackprop.ann.OutputLayer;
import org.jbackprop.ann.OutputNeuron;

public abstract class LossFunction {
    public abstract double calculateLoss(double loss);
    public abstract double derivative(double loss);
    public abstract double calculateSumLoss(OutputLayer outputLayer);
    public double calculateActualSumLoss(OutputLayer outputLayer){
        double sumLoss =0;
        var neurons = outputLayer.getNeurons();

        for (OutputNeuron neuron : neurons){
            sumLoss += neuron.calculateLoss();
        }
        return sumLoss;
    }
}
