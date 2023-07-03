package org.jbackprop.ann.lossfunctions;

import org.jbackprop.ann.OutputLayer;
import org.jbackprop.ann.OutputNeuron;

public class MSE implements LossFunction {
    @Override
    public double calculateLoss(double loss){
        return calculateMSE(loss);
    }

    @Override
    public double derivative(double loss){
        return calculateMSEDerivative(loss);
    }

    @Override
    public double calculateSumLoss(OutputLayer outputLayer) {
        double sumLoss =0;
        var neurons = outputLayer.getNeurons();

        for (OutputNeuron neuron : neurons){
            sumLoss += calculateLoss(neuron.getActualLoss());
        }
        return sumLoss;
    }

    private static double calculateMSE(double loss){
        return loss*loss;
    }

    private static double calculateMSEDerivative(double loss){
        return 2 * loss;
    }
}
