package org.jbackprop.ann;

import lombok.Getter;
import lombok.Setter;

@Setter @Getter
public class NetworkParams<T extends Neuron> {
    private Class<T> neuronClass;
    private LossFunction lossFunction;
    private double learningRate;
    private int epochs;
    private double desiredLoss;

    public NetworkParams(Class<T> neuronClass,
                         LossFunction lossFunction,
                         double learningRate,
                         int epochs,
                         double desiredLoss){
        this.neuronClass = neuronClass;
        this.lossFunction = lossFunction;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.desiredLoss = desiredLoss;
    }

    public NetworkParams(){
        this((Class<T>) SigmoidNeuron.class,
                new MSE(),
                0.1,
                1000000000,
                0.01);
    }
}
