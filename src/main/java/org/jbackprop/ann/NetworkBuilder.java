package org.jbackprop.ann;

import lombok.Getter;
import org.jbackprop.ann.activationfunctions.*;
import org.jbackprop.ann.lossfunctions.LossFunction;
import org.jbackprop.ann.lossfunctions.LossFunctions;
import org.jbackprop.ann.lossfunctions.MSE;
import org.jbackprop.dataset.DataSet;

@Getter
public class NetworkBuilder {
    private int[] architecture;
    private ActivationFunction activationFunction = new Tanh();
    private LossFunction lossFunction = new MSE();
    private double learningRate = 0.1;
    private double momentum;
    private int epochs = 1000000000;
    private double desiredLoss = 0.01;
    private DataSet dataSet;
    private Network network = new Network();

    public NetworkBuilder activationFunction(ActivationFunctions activationFunctions) {
        activationFunction = switch (activationFunctions){
            case SIGMOID -> new Sigmoid();
            case TANH -> new Tanh();
            case RELU -> new Relu();
        };
        return this;
    }

    public NetworkBuilder lossFunction(LossFunctions lossFunctions) {

        lossFunction = switch (lossFunctions){
            case MSE -> new MSE();
        };
        return this;
    }

    public NetworkBuilder learningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public NetworkBuilder epochs(int epochs) {
        this.epochs = epochs;
        return this;
    }

    public NetworkBuilder desiredLoss(double desiredLoss) {
        this.desiredLoss = desiredLoss;
        return this;
    }

    public NetworkBuilder dataSet(DataSet dataSet) {
        this.dataSet = dataSet;
        return this;
    }

    public NetworkBuilder architecture(int... architecture) {
        this.architecture = architecture;
        return this;
    }

    public NetworkBuilder network(Network network) {
        this.network = network;
        return this;
    }

    public NetworkBuilder momentum(double momentum) {
        this.momentum = momentum;
        return this;
    }

    public Network build() {
        network.setNetworkBuilder(this);
        network.build();

        return network;
    }

}
