package org.jbackprop.ann;

import org.jbackprop.dataset.DataSet;
import org.jbackprop.dataset.Row;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;
@Getter @Setter
public class Network{
    private List<Layer> layers;
    private Layer outputLayer;
    private double lossOfEpoch;
    private double lossOfPreviousEpoch;

    private final int[] neuronLayout;
    private double[] networkOutput;
    private LossFunction lossFunction;
    private NetworkParams networkParams;
    private DataSet dataSet;

    private Network(NetworkParams networkParams,
                   DataSet dataSet, int... neuronLayout){
       this(networkParams, neuronLayout);
       this.dataSet = dataSet;

        build();
        learn();
    }

    public Network(NetworkParams networkParams,
                   int... neuronLayout){
        this.neuronLayout = neuronLayout;
        this.networkParams = networkParams;
        lossFunction = networkParams.getLossFunction();
    }

    public Network(DataSet dataSet, int... neuronLayout){
        this(new NetworkParams(), dataSet, neuronLayout);
    }

    public void beforeEpoch(){
    }
    public void afterEpoch(){
        
    }

    private void build(){
        layers = new ArrayList<>();
        var numConnections = dataSet.getInputDimension();
        Layer previousLayer = null;

        for(int numNeurons: neuronLayout){
            var layer = new Layer();

            previousLayer = layer.build(numNeurons,
                    numConnections, previousLayer, networkParams);

            numConnections = numNeurons;

            layers.add(layer);
        }

        outputLayer = layers.get(layers.size()-1);
    }

    public void forwardPass(double[] firstInput){
        double[] previousActivations = firstInput;

        for(Layer layer: layers)
            previousActivations =
                    layer.calculateActivations(previousActivations);

        networkOutput = outputLayer.getActivations();
    }

    public void learn(){
        var epochs = networkParams.getEpochs();
        var desiredLoss = networkParams.getDesiredLoss();

        do{
            beforeEpoch();
            epoch(dataSet);
            afterEpoch();
            lossOfPreviousEpoch = lossOfEpoch;
            lossOfEpoch =0;
            epochs--;
        }

        while (epochs>0 &&
                desiredLoss<lossOfPreviousEpoch);
    }

    private void epoch(DataSet dataSet){
        for(Row row: dataSet.getRows()){
            forwardPass(row.getInput());
            backwardPass(row.getTarget());
            lossOfEpoch += calculateLossOfIteration();
            adjustWeights();
        }
    }

    private void backwardPass(double[] targets){
        outputLayer.calculateDeltas(targets);

        for (int i=layers.size()-2; i>=0; i--){
            layers.get(i).calculateDeltas();
        }
    }

    private void adjustWeights(){
        for (Layer layer: layers){
            layer.adjustWeights();
        }
    }

    private double calculateLossOfIteration(){
        return lossFunction.calculateSumLoss(outputLayer);
    }
}
