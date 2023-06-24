package org.jbackprop;

import java.util.ArrayList;
import java.util.List;

public class Network {
    private final List<Layer> layers = new ArrayList<>();
    private final int numLayers;
    private float learningRate;

    private final int[] neuronLayout;
    private List<Double> networkOutput;

    public Network(float learningRate, int... neuronLayout) {
        this.neuronLayout = neuronLayout;
        numLayers = neuronLayout.length;
        this.learningRate = learningRate;
    }

    private void build(int inputDimension){

        var numConnections = inputDimension;

        for(int numNeurons: neuronLayout){
            layers.add(new Layer(numNeurons, numConnections));

            numConnections = numNeurons;
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
