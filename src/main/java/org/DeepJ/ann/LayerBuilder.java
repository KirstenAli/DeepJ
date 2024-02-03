package org.DeepJ.ann;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;

public class LayerBuilder{
    public static <T extends Neuron> List<T> build(int numNeurons, int connectionsPerNeuron,
                                                   HiddenLayer previousLayer, NetworkBuilder networkBuilder,
                                                   Class<T> neuronClass) {
        var neurons = new ArrayList<T>();
        try{
            for (int i = 0; i < numNeurons; i++) {

                Constructor<T> constructor = neuronClass.getDeclaredConstructor(Integer.class,
                        HiddenLayer.class,
                        NetworkBuilder.class);

                T neuron = constructor.newInstance(connectionsPerNeuron, previousLayer, networkBuilder);
                neuron.buildConnections();
                neurons.add(neuron);
            }
        } catch (InvocationTargetException |
                 NoSuchMethodException |
                 InstantiationException |
                 IllegalAccessException e) {
            throw new RuntimeException(e);
        }

        return neurons;
    }
}
