package org.DeepJ.ann;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.DeepJ.ann.lossfunctions.LossFunction;
import org.DeepJ.dataset.DataSet;
import org.DeepJ.dataset.Row;
import org.DeepJ.persistence.PersistenceManager;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

@Getter @Setter @NoArgsConstructor
public class Network implements Serializable {
    @JsonProperty
    private List<HiddenLayer> hiddenLayers;
    @JsonProperty
    private OutputLayer outputLayer;
    private int currentEpoch;
    private double lossOfEpoch;
    private double lossOfPreviousEpoch;
    @JsonProperty
    private int[] architecture;
    private double[] networkOutput;
    private LossFunction lossFunction;
    private NetworkBuilder networkBuilder;
    private DataSet dataSet;

    public void setNetworkBuilder(NetworkBuilder networkBuilder) {
        this.networkBuilder = networkBuilder;
        architecture = networkBuilder.getArchitecture();
        dataSet = networkBuilder.getDataSet();
        lossFunction = networkBuilder.getLossFunction();
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

    public void forwardPass(double[] input) {
        double[] previousActivations = input;

        for (HiddenLayer layer : hiddenLayers)
            previousActivations =
                    layer.applyActivations(previousActivations);

        networkOutput = outputLayer.applyActivations(previousActivations);
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

    private void epoch(DataSet dataSet) {
        for (Row row : dataSet.getRows()) {
            forwardPass(row.getInputs());
            backwardPass(row.getTargets());
            lossOfEpoch += calculateLossOfIteration();
            adjustWeights();
        }
    }

    private void backwardPass(double[] targets) {
        outputLayer.calculateDeltas(targets);

        for (int i = hiddenLayers.size() - 1; i >= 0; i--) {
            hiddenLayers.get(i).calculateDeltas();
        }
    }

    private void adjustWeights() {
        outputLayer.adjustWeights();

        for (HiddenLayer layer : hiddenLayers) {
            layer.adjustWeights();
        }
    }

    private double calculateLossOfIteration() {
        return lossFunction.calculateLossOfIteration(outputLayer);
    }

    public void save(String filePath){
        PersistenceManager.saveNetwork(this, filePath);
    }

    public void saveWeightsAsJsonFile(String filePath){
        PersistenceManager.saveWeightsAsJsonFile(this, filePath);
    }
}
