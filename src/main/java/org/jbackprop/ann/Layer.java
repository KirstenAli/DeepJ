package org.jbackprop.ann;

import com.fasterxml.jackson.annotation.JsonIgnore;

import lombok.Getter;
import lombok.Setter;

import java.util.List;

@Setter
@Getter
public abstract class Layer<T extends Neuron> {
    protected List<T> neurons;
    @JsonIgnore
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

    public void adjustWeights() {
        for (Neuron neuron : neurons)
            neuron.adjustWeights();
    }
}
