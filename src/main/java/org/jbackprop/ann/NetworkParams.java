package org.jbackprop.ann;

import lombok.Getter;
import lombok.Setter;

@Setter @Getter
public class NetworkParams {
    private Class neuronClass;
    private LossFunction lossFunction;
    private double learningRate;
    private int epochs;
    private double desiredLoss;

    public <T extends Neuron> NetworkParams(Class<T> neuronClass,
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
        this(SigmoidNeuron.class);
    }

    public <T extends Neuron> NetworkParams(Class<T> neuronClass){
        this(neuronClass,
                new MSE(),
                0.1,
                1000000000,
                0.01);
    }
}
