package org.jbackprop.ann;

import lombok.Getter;
import lombok.Setter;
import org.jbackprop.ann.lossfunctions.LossFunction;

@Setter
@Getter
public class OutputNeuron extends Neuron {

    private double loss;
    private double actualLoss;
    private final LossFunction lossFunction;

    public OutputNeuron(Integer numConnections, HiddenLayer previousLayer, NetworkBuilder networkBuilder) {
        super(numConnections, previousLayer, networkBuilder);
        this.lossFunction = networkBuilder.getLossFunction();
    }

    private double calculateActualLoss(double target) {
        actualLoss = target - activation;
        return actualLoss;
    }

    public double calculateLoss() {
        loss = lossFunction.calculateLoss(actualLoss);
        return loss;
    }

    public void calculateDelta(double target) {
        var actualLoss = calculateActualLoss(target);
        var lossDerivative = lossFunction.derivative(actualLoss);
        var activationDerivative = activationFunction.derivative(net, activation);

        delta = activationDerivative * lossDerivative;
    }

}
