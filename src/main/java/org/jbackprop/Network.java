package org.jbackprop;

import java.util.ArrayList;
import java.util.List;

public class Network {
    private final List<Layer> layers = new ArrayList<>();
    private final int numLayers;
    private final float learningRate;

    private final int[] neuronLayout;
    private List<Double> networkOutput;

    public Network(float learningRate, int... neuronLayout) {
        this.neuronLayout = neuronLayout;
        numLayers = neuronLayout.length;
        this.learningRate = learningRate;
    }

    private void build(int inputDimension,
                       Class<Neuron> neuronClass){

        var numConnections = inputDimension;
        Layer previousLayer = null;

        for(int numNeurons: neuronLayout){
            var layer = new Layer();

            previousLayer = layer.build(numNeurons,
                    numConnections,
                    previousLayer,
                    neuronClass);

            numConnections = numNeurons;

            layers.add(layer);
        }
    }

    public List<Double> forwardPass(List<Double> firstInput){
        List<Double> previousActivations = firstInput;

        for(Layer layer: layers)
            previousActivations =
                    layer.calculateActivations(previousActivations);

        networkOutput = layers.get(numLayers-1).getActivations();

        return networkOutput;
    }
}
