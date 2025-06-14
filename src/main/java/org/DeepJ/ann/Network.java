package org.DeepJ.ann;

import org.DeepJ.ann.lossfunctions.LossFunction;
import org.DeepJ.ann.dataset.DataSet;
import org.DeepJ.ann.dataset.Row;
import org.DeepJ.persistence.PersistenceManager;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class Network implements Serializable {
    private List<HiddenLayer> hiddenLayers;
    private OutputLayer outputLayer;
    private int currentEpoch;
    private double lossOfEpoch;
    private double lossOfPreviousEpoch;
    private int[] architecture;
    private double[] output;
    private LossFunction lossFunction;
    private NetworkBuilder networkBuilder;
    private DataSet dataSet;

    public void setNetworkBuilder(NetworkBuilder networkBuilder) {
        this.networkBuilder = networkBuilder;
        architecture = networkBuilder.getArchitecture();
        dataSet = networkBuilder.getDataSet();
        lossFunction = networkBuilder.getLossFunction();
    }

    public Network() {
    }

    public void build() {
        hiddenLayers = new ArrayList<>();
        var connectionsPerNeuron = dataSet.getInputDimension();
        HiddenLayer previousLayer = null;

        int i;
        for (i = 0; i < architecture.length - 1; i++) {
            var hiddenLayer = new HiddenLayer();
            hiddenLayer.build(architecture[i],
                    connectionsPerNeuron, previousLayer, networkBuilder);

            connectionsPerNeuron = architecture[i];
            previousLayer = hiddenLayer;

            hiddenLayers.add(hiddenLayer);
        }

        outputLayer = new OutputLayer();
        outputLayer.build(architecture[i],
                connectionsPerNeuron, previousLayer, networkBuilder);
    }

    public void forward(double[] input) {
        double[] previousActivations = input;

        for (HiddenLayer layer : hiddenLayers)
            previousActivations =
                    layer.applyActivations(previousActivations);

        output = outputLayer.applyActivations(previousActivations);
    }

    public void learn() {
        var epochs = networkBuilder.getEpochs();
        var desiredLoss = networkBuilder.getDesiredLoss();
        var beforeEpoch = networkBuilder.getBeforeEpoch();
        var afterEpoch = networkBuilder.getAfterEpoch();

        do {
            currentEpoch++;
            beforeEpoch.perform(this);
            epoch(dataSet);
            afterEpoch.perform(this);
            lossOfPreviousEpoch = lossOfEpoch;
            lossOfEpoch = 0;
            epochs--;
        }

        while (epochs > 0 &&
                desiredLoss < lossOfPreviousEpoch);
    }

    public void epoch(DataSet dataSet) {
        for (Row row : dataSet.getRows()) {
            forward(row.getInputs());
            backward(row.getTargets());
            lossOfEpoch += calculateLoss();
            updateWeights();
        }
    }

    public void backward(double[] targets) {
        outputLayer.calculateDeltas(targets);

        for (int i = hiddenLayers.size() - 1; i >= 0; i--) {
            hiddenLayers.get(i).calculateDeltas();
        }
    }

    public void updateWeights() {
        outputLayer.updateWeights();
        for (HiddenLayer layer : hiddenLayers) {
            layer.updateWeights();
        }
    }

    public double calculateLoss() {
        return lossFunction.calculateLossOfIteration(outputLayer);
    }

    public double[] getInputGradient() {
        return hiddenLayers.get(0).neurons.stream()
                .mapToDouble(HiddenNeuron::sumWeightedDeltasFromInputs)
                .toArray();
    }

    public void save(String filePath){
        PersistenceManager.saveNetwork(this, filePath);
    }

    public List<HiddenLayer> getHiddenLayers() {
        return hiddenLayers;
    }

    public void setHiddenLayers(List<HiddenLayer> hiddenLayers) {
        this.hiddenLayers = hiddenLayers;
    }

    public OutputLayer getOutputLayer() {
        return outputLayer;
    }

    public void setOutputLayer(OutputLayer outputLayer) {
        this.outputLayer = outputLayer;
    }

    public int getCurrentEpoch() {
        return currentEpoch;
    }

    public void setCurrentEpoch(int currentEpoch) {
        this.currentEpoch = currentEpoch;
    }

    public double getLossOfEpoch() {
        return lossOfEpoch;
    }

    public void setLossOfEpoch(double lossOfEpoch) {
        this.lossOfEpoch = lossOfEpoch;
    }

    public double getLossOfPreviousEpoch() {
        return lossOfPreviousEpoch;
    }

    public void setLossOfPreviousEpoch(double lossOfPreviousEpoch) {
        this.lossOfPreviousEpoch = lossOfPreviousEpoch;
    }

    public int[] getArchitecture() {
        return architecture;
    }

    public void setArchitecture(int[] architecture) {
        this.architecture = architecture;
    }

    public double[] getOutput() {
        return output;
    }

    public void setOutput(double[] output) {
        this.output = output;
    }

    public LossFunction getLossFunction() {
        return lossFunction;
    }

    public void setLossFunction(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }

    public NetworkBuilder getNetworkBuilder() {
        return networkBuilder;
    }

    public DataSet getDataSet() {
        return dataSet;
    }

    public void setDataSet(DataSet dataSet) {
        this.dataSet = dataSet;
    }
}
