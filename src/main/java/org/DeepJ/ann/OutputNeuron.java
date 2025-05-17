package org.DeepJ.ann;

import org.DeepJ.ann.lossfunctions.LossFunction;

public class OutputNeuron extends Neuron {
    private double actualLoss;
    private LossFunction lossFunction;

    public OutputNeuron(Integer numConnections, HiddenLayer previousLayer, NetworkBuilder networkBuilder) {
        super(numConnections, previousLayer, networkBuilder);
        this.lossFunction = networkBuilder.getLossFunction();
    }

    public void calculateDelta(double target) {
        actualLoss = lossFunction.calculateActualLoss(target, activation);
        var lossDerivative = lossFunction.derivative(actualLoss);
        var activationDerivative = activationFunction.derivative(net, activation);

        delta = activationDerivative * lossDerivative;
    }

    public double getActualLoss() {
        return actualLoss;
    }

    public void setActualLoss(double actualLoss) {
        this.actualLoss = actualLoss;
    }

    public LossFunction getLossFunction() {
        return lossFunction;
    }

    public void setLossFunction(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }
}
