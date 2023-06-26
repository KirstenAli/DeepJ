package org.jbackprop;

import lombok.Getter;
import lombok.Setter;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;

@Getter @Setter
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

    public Layer build(int numNeurons, int numConnections,
                       Layer previousLayer, GlobalParams globalParams){
        for(int i=0; i<numNeurons; i++){
            try {
                Class<Neuron> neuronClass = globalParams.getNeuronClass();
                Constructor<Neuron> constructor = neuronClass.getDeclaredConstructor();
                Neuron neuron = constructor.newInstance(numConnections, previousLayer, globalParams);

                neurons.add(neuron);

            } catch (InvocationTargetException |
                     NoSuchMethodException |
                     InstantiationException |
                     IllegalAccessException e) {
                throw new RuntimeException(e);
            }
        }
        return this;
    }
}
