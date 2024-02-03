package org.DeepJ.ann.lossfunctions;

import org.DeepJ.ann.OutputLayer;
import org.DeepJ.ann.OutputNeuron;

import java.io.Serializable;

public abstract class LossFunction implements Serializable {
    public abstract double calculateActualLoss(double target, double actual);
    public abstract double calculateLoss(double actualLoss);
    public abstract double derivative(double actualLoss);

    public double calculateLossOfIteration(OutputLayer outputLayer) {
        double sumLoss =0;
        var neurons = outputLayer.getNeurons();

        for (OutputNeuron neuron : neurons){
            sumLoss += calculateLoss(neuron.getActualLoss());
        }
        return sumLoss/outputLayer.getSize();
    }
}
