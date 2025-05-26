package org.DeepJ.annoop;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.io.Serializable;
import java.util.List;

public abstract class Layer<T extends Neuron> implements Serializable {
    @JsonProperty
    protected List<T> neurons;
    private double[] activations;

    public double[] applyActivations(double[] inputs) {
        setInputs(inputs);

        activations = new double[neurons.size()];

        for (int i = 0; i < neurons.size(); i++) {
            activations[i] = neurons.get(i).applyActivation();
        }
        return activations;
    }

    private void setInputs(double[] inputs) {
        for (Neuron neuron : neurons) {
            neuron.setInputs(inputs);
        }
    }

    abstract void build(int numNeurons,
                        int connectionsPerNeuron,
                        HiddenLayer previousLayer,
                        NetworkBuilder networkBuilder);

    public void updateWeights() {
        for (Neuron neuron : neurons)
            neuron.updateWeights();
    }

    public List<T> getNeurons() {
        return neurons;
    }

    public void setNeurons(List<T> neurons) {
        this.neurons = neurons;
    }

    public double[] getActivations() {
        return activations;
    }

    public void setActivations(double[] activations) {
        this.activations = activations;
    }
}
