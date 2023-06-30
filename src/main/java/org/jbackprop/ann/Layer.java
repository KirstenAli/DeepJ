package org.jbackprop.ann;

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

    public <T extends Neuron> Layer build(int numNeurons, int numConnections,
                       Layer previousLayer, NetworkParams<T> networkParams){
        for(int i=0; i<numNeurons; i++){
            try {
                Class<T> neuronClass = networkParams.getNeuronClass();

                Constructor<T> constructor = neuronClass.getDeclaredConstructor(Integer.class,
                        Layer.class,
                        NetworkParams.class);

                Neuron neuron = constructor.newInstance(numConnections, previousLayer, networkParams);

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

    public void calculateDeltas(){
        for(Neuron neuron: neurons)
            neuron.calculateDelta();
    }

    public void calculateDeltas(List<Double> targets){
        for(int i=0; i<targets.size(); i++)
            neurons.get(i).calculateDelta(targets.get(i));
    }

    public void adjustWeights(){
        for(Neuron neuron: neurons)
            neuron.adjustWeights();
    }
}
