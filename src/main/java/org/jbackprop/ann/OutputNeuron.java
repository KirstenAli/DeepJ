package org.jbackprop.ann;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.Getter;
import lombok.Setter;
import org.jbackprop.ann.lossfunctions.LossFunction;

@Setter
@Getter
public class OutputNeuron extends Neuron {

    private double actualLoss;
    @JsonIgnore
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

}
