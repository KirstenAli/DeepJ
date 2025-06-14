package org.DeepJ.ann;

import org.DeepJ.ann.activationfunctions.*;
import org.DeepJ.ann.lossfunctions.LossFunction;
import org.DeepJ.ann.lossfunctions.LossFunctions;
import org.DeepJ.ann.lossfunctions.MSE;
import org.DeepJ.ann.dataset.DataSet;

import java.io.Serializable;

public class NetworkBuilder implements Serializable {
    private int[] architecture;
    private ActivationFunction activationFunction = new Tanh();
    private LossFunction lossFunction = new MSE();
    private double learningRate = 0.1;
    private double momentum;
    private int epochs = 1000000000;
    private double desiredLoss = 0.01;
    private DataSet dataSet;
    private transient EpochOperation beforeEpoch;
    private transient EpochOperation afterEpoch;
    private Network network;

    public NetworkBuilder() {
        network = new Network();
        beforeEpoch = network->{};
        afterEpoch= network->{};
    }

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

    public NetworkBuilder beforeEpoch(EpochOperation beforeEpoch) {
        this.beforeEpoch = beforeEpoch;
        return this;
    }

    public NetworkBuilder afterEpoch(EpochOperation afterEpoch) {
        this.afterEpoch = afterEpoch;
        return this;
    }

    public int[] getArchitecture() {
        return architecture;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public LossFunction getLossFunction() {
        return lossFunction;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public double getMomentum() {
        return momentum;
    }

    public int getEpochs() {
        return epochs;
    }

    public double getDesiredLoss() {
        return desiredLoss;
    }

    public DataSet getDataSet() {
        return dataSet;
    }

    public EpochOperation getBeforeEpoch() {
        return beforeEpoch;
    }

    public EpochOperation getAfterEpoch() {
        return afterEpoch;
    }

    public Network getNetwork() {
        return network;
    }
}
