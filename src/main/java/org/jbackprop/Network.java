package org.jbackprop;

import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;
@Getter @Setter
public class Network{
    private final List<Layer> layers;
    private final int numLayers;
    private final float learningRate;

    private final int[] neuronLayout;
    private List<Double> networkOutput;
    private LossFunction lossFunction;

    public Network(float learningRate, int... neuronLayout){
        this.neuronLayout = neuronLayout;
        numLayers = neuronLayout.length;
        this.learningRate = learningRate;
        layers = new ArrayList<>();
    }

    public void beforeEpoch(){
    }
    public void AfterEpoch(){
    }

    private void build(int inputDimension,GlobalParams globalParams){

        var numConnections = inputDimension;
        Layer previousLayer = null;

        for(int numNeurons: neuronLayout){
            var layer = new Layer();

            previousLayer = layer.build(numNeurons,
                    numConnections, previousLayer, globalParams);

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

    private double calculateSumError(){
        var outputLayer = layers.get(numLayers-1);
        return lossFunction.calculateSumError(outputLayer);
    }
}
