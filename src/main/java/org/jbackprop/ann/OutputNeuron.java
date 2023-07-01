package org.jbackprop.ann;

import lombok.Getter;
import lombok.Setter;

@Setter @Getter
public class OutputNeuron extends Neuron{

    private double loss;
    private double actualLoss;
    private final LossFunction lossFunction;

    public OutputNeuron(Integer numConnections, HiddenLayer previousLayer, NetworkBuilder networkBuilder) {
        super(numConnections, previousLayer, networkBuilder);
        this.lossFunction = networkBuilder.getLossFunction();
    }

    private double dLoss(double target){
        return lossFunction.dLoss(calculateActualLoss(target));
    }

    private double calculateActualLoss(double target){
        actualLoss = target-activation;
        return actualLoss;
    }

    public double calculateLoss(){
        loss = lossFunction.calculateLoss(actualLoss);
        return loss;
    }

    public void calculateDelta(double target){
        var dLoss = dLoss(target);
        var activationDerivative = activationFunction.derivative(net,activation);
        delta = activationDerivative*dLoss;
    }
}
