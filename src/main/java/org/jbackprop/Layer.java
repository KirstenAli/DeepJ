package org.jbackprop;

import lombok.Getter;

import java.util.ArrayList;
import java.util.List;

@Getter
public class Layer {
    private final List<Neuron> neurons = new ArrayList<>();
    private final List<Double> activations = new ArrayList<>();

    public List<Double> calculateActivations(){
        for(Neuron neuron: neurons){
            activations.add(neuron.calculateActivation());
        }

        return activations;
    }

    public void setInputs(List<Double> inputs){
        for (Neuron neuron: neurons){
            neuron.setInputs(inputs);
        }
    }

    public void build(int numNeurons, int numConnections){
        for(int i=0; i<numNeurons; i++){
            neurons.add(new Neuron(numConnections));
        }
    }

}
