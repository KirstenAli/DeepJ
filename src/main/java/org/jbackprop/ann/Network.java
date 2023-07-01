package org.jbackprop.ann;

import org.jbackprop.dataset.DataSet;
import org.jbackprop.dataset.Row;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;
@Getter @Setter
public class Network{
    private List<HiddenLayer> hiddenLayers;
    private OutputLayer outputLayer;
    private double lossOfEpoch;
    private double lossOfPreviousEpoch;

    private int[] neuronLayout;
    private double[] networkOutput;
    private LossFunction lossFunction;
    private NetworkBuilder networkBuilder;
    private DataSet dataSet;

    public void setNetworkBuilder(NetworkBuilder networkBuilder) {
        this.neuronLayout = networkBuilder.getNeuronLayout();
        this.dataSet = networkBuilder.getDataSet();
        this.networkBuilder = networkBuilder;
        lossFunction = networkBuilder.getLossFunction();
    }

    public Network() {
    }

    public void beforeEpoch(){
    }
    public void afterEpoch(){
        
    }

    public void build(){
        hiddenLayers = new ArrayList<>();
        var connectionsPerNeuron = dataSet.getInputDimension();
        HiddenLayer previousLayer = null;

        int i;
        for(i=0; i<neuronLayout.length-1; i++){
            var hiddenLayer = new HiddenLayer();
            hiddenLayer.build(neuronLayout[i],
                    connectionsPerNeuron, previousLayer, networkBuilder);

            connectionsPerNeuron = neuronLayout[i];
            previousLayer = hiddenLayer;

            hiddenLayers.add(hiddenLayer);
        }

        outputLayer = new OutputLayer();
        outputLayer.build(neuronLayout[i],
                connectionsPerNeuron, previousLayer, networkBuilder);
    }

    public void forwardPass(double[] firstInput){
        double[] previousActivations = firstInput;

        for(HiddenLayer layer: hiddenLayers)
            previousActivations =
                    layer.calculateActivations(previousActivations);


        networkOutput = outputLayer.calculateActivations(previousActivations);
    }

    public void learn(){
        var epochs = networkBuilder.getEpochs();
        var desiredLoss = networkBuilder.getDesiredLoss();

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

        for (int i = hiddenLayers.size()-1; i>=0; i--){
            hiddenLayers.get(i).calculateDeltas();
        }
    }

    private void adjustWeights(){
        outputLayer.adjustWeights();

        for (HiddenLayer layer: hiddenLayers){
            layer.adjustWeights();
        }
    }

    private double calculateLossOfIteration(){
        return lossFunction.calculateSumLoss(outputLayer);
    }
}
