package org.jbackprop;

import java.util.ArrayList;
import java.util.List;

public class Network {
    private final List<Layer> layers = new ArrayList<>();
    private final int numLayers;
    private final float learningRate;

    private final int[] neuronLayout;
    private List<Double> networkOutput;
    private LossFunction lossFunction;

    public Network(float learningRate, int... neuronLayout) {
        this.neuronLayout = neuronLayout;
        numLayers = neuronLayout.length;
        this.learningRate = learningRate;
    }

    private void build(int inputDimension,
                       Class<Neuron> neuronClass,
                       LossFunction lossFunction){

        var numConnections = inputDimension;
        Layer previousLayer = null;

        for(int numNeurons: neuronLayout){
            var layer = new Layer();

            previousLayer = layer.build(numNeurons,
                    numConnections, previousLayer,
                    neuronClass, lossFunction);

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

    public double calculateSumError(List<Double> target){
        var outputLayer = layers.get(numLayers-1);
        return lossFunction.calculateSumError(outputLayer, target);
    }
}
