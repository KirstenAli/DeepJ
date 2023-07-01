package org.jbackprop.ann;

import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;

@Getter @Setter
public class Layer {
    private final List<Neuron> neurons = new ArrayList<>();
    private double[] activations;

    public double[] calculateActivations(double[] inputs){
        setInputs(inputs);

        activations = new double[neurons.size()];

        for(int i=0; i<neurons.size(); i++){
            activations[i] = neurons.get(i).calculateActivation();
        }
        return activations;
    }

    private void setInputs(double[] inputs){
        for (Neuron neuron: neurons){
            neuron.setInputs(inputs);
        }
    }

    public Layer build(int numNeurons, int connectionsPerNeuron,
                       Layer previousLayer, NetworkParams networkParams){

        for(int i=0; i<numNeurons; i++){
            Neuron neuron = new Neuron(connectionsPerNeuron,
                    previousLayer,
                    networkParams);

            neurons.add(neuron);
        }
        return this;
    }

    public void calculateDeltas(){
        for(Neuron neuron: neurons)
            neuron.calculateDelta();
    }

    public void calculateDeltas(double[] targets){
        for(int i=0; i<targets.length; i++)
            neurons.get(i).calculateDelta(targets[i]);
    }

    public void adjustWeights(){
        for(Neuron neuron: neurons)
            neuron.adjustWeights();
    }
}
