package org.jbackprop.ann;

import lombok.Getter;
import lombok.Setter;

import java.util.List;

@Getter @Setter
public abstract class Layer <T extends Neuron> {
    protected List<T> neurons;
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

    abstract void build(int numNeurons,
                        int connectionsPerNeuron,
                        HiddenLayer previousLayer,
                        NetworkBuilder networkBuilder);

    public void adjustWeights(){
        for(Neuron neuron: neurons)
            neuron.adjustWeights();
    }
}
