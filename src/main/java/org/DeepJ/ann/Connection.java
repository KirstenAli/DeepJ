package org.DeepJ.ann;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.Setter;

import java.io.Serializable;

@Setter
@Getter
public class Connection implements Serializable {
    private Neuron outputNeuron;
    private Neuron inputNeuron;
    private double input;
    @JsonProperty
    private double weight;
    private double product;
    private double learningRate;
    private double momentum;
    private double prevUpdate;
    private NetworkBuilder networkBuilder;

    public Connection(NetworkBuilder networkBuilder) {
        weight = Math.random() - 0.5;
        input = 1;
        learningRate = networkBuilder.getLearningRate();
        momentum = networkBuilder.getMomentum();
    }

    public double calculateProduct() {
        product = input * weight;
        return product;
    }

    public double calculateWeightedDelta() {
        var delta = inputNeuron.getDelta();

        return delta * weight;
    }

    public void adjustWeight() {
        var delta = inputNeuron.getDelta();
        var update = learningRate * delta * input;
        update+= momentum*prevUpdate; // add momentum to update
        weight += update;
        prevUpdate = update;
    }

    public Neuron getOutputNeuron() {
        return outputNeuron;
    }

    public void setOutputNeuron(Neuron outputNeuron) {
        this.outputNeuron = outputNeuron;
    }

    public Neuron getInputNeuron() {
        return inputNeuron;
    }

    public void setInputNeuron(Neuron inputNeuron) {
        this.inputNeuron = inputNeuron;
    }

    public double getInput() {
        return input;
    }

    public void setInput(double input) {
        this.input = input;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public double getProduct() {
        return product;
    }

    public void setProduct(double product) {
        this.product = product;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public double getMomentum() {
        return momentum;
    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    public double getPrevUpdate() {
        return prevUpdate;
    }

    public void setPrevUpdate(double prevUpdate) {
        this.prevUpdate = prevUpdate;
    }

    public NetworkBuilder getNetworkBuilder() {
        return networkBuilder;
    }

    public void setNetworkBuilder(NetworkBuilder networkBuilder) {
        this.networkBuilder = networkBuilder;
    }
}
