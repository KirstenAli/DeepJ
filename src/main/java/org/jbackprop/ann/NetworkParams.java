package org.jbackprop.ann;

import lombok.Getter;
import lombok.Setter;

@Setter @Getter
public class NetworkParams {
    private ActivationFunction activationFunction;
    private LossFunction lossFunction;
    private double learningRate;
    private int epochs;
    private double desiredLoss;

    public NetworkParams(ActivationFunction activationFunction,
                         LossFunction lossFunction,
                         double learningRate,
                         int epochs,
                         double desiredLoss){
        this.activationFunction = activationFunction;
        this.lossFunction = lossFunction;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.desiredLoss = desiredLoss;
    }

    public NetworkParams(){
        this(new Sigmoid());
    }

    public NetworkParams(ActivationFunction activationFunction){
        this(activationFunction,
                new MSE(),
                0.1,
                1000000000,
                0.01);
    }
}
