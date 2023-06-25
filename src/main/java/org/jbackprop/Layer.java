package org.jbackprop;

import lombok.Getter;

import java.util.ArrayList;
import java.util.List;

@Getter
public class Layer {
    private final List<Neuron> neurons = new ArrayList<>();
    private List<Double> activations;

    public List<Double> calculateActivations(List<Double> inputs){
        setInputs(inputs);

        activations = new ArrayList<>();

        for(Neuron neuron: neurons){
            activations.add(neuron.calculateActivation());
        }

        return activations;
    }

    private void setInputs(List<Double> inputs){
        for (Neuron neuron: neurons){
            neuron.setInputs(inputs);
        }
    }

    public Layer build(int numNeurons,
                       int numConnections,
                       Layer previousLayer){
        for(int i=0; i<numNeurons; i++)
            neurons.add(new Neuron(numConnections, previousLayer));

        return this;
    }

}
