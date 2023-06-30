package org.jbackprop.ann;

import lombok.Getter;
import lombok.Setter;

@Setter @Getter
public class NetworkParams<T extends Neuron> {
    private Class<T> neuronClass;
    private LossFunction lossFunction;
    private double learningRate;
    private int epochs;
    private double desiredLoss = 0.01;

    public NetworkParams(Class<T> neuronClass,
                         LossFunction lossFunction,
                         double learningRate,
                         int epochs) {
        this.neuronClass = neuronClass;
        this.lossFunction = lossFunction;
        this.learningRate = learningRate;
        this.epochs = epochs;
    }
}
