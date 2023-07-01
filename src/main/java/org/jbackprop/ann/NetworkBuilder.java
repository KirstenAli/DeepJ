package org.jbackprop.ann;

import lombok.Getter;
import org.jbackprop.dataset.DataSet;

@Getter
public class NetworkBuilder {
    private int[] neuronLayout;
    private ActivationFunction activationFunction = new Sigmoid();
    private LossFunction lossFunction = new MSE();
    private double learningRate = 0.1;
    private int epochs = 1000000000;
    private double desiredLoss = 0.01;
    private DataSet dataSet;
    private Network network = new Network();

    public NetworkBuilder activationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
        return this;
    }

    public NetworkBuilder lossFunction(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
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

    public NetworkBuilder neuronLayout(int... neuronLayout) {
        this.neuronLayout = neuronLayout;
        return this;
    }

    public NetworkBuilder network(Network network) {
        this.network = network;
        return this;
    }

    public Network build(){
        network.setNetworkBuilder(this);
        network.build();

        return network;
    }
}
