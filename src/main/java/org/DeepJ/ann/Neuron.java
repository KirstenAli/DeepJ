package org.DeepJ.ann;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.Setter;
import org.DeepJ.ann.activationfunctions.ActivationFunction;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

@Setter
@Getter
public abstract class Neuron implements Serializable {
    protected double net;
    protected double activation;
    protected double delta;
    @JsonProperty
    protected List<Connection> inputConnections;
    protected ActivationFunction activationFunction;
    private int numConnections;
    private HiddenLayer previousLayer;
    private NetworkBuilder networkBuilder;

    public Neuron(Integer numConnections,
                  HiddenLayer previousLayer,
                  NetworkBuilder networkBuilder) {

        this.activationFunction = networkBuilder.getActivationFunction();
        this.numConnections = numConnections;
        this.previousLayer = previousLayer;
        this.networkBuilder = networkBuilder;

        inputConnections = new ArrayList<>();
    }

    public double applyActivation() {
        var net = calculateNet();
        activation = activationFunction.applyActivation(net);
        return activation;
    }

    public double calculateNet() {
        net=0;
        for (Connection connection : inputConnections)
            net += connection.calculateProduct();
        return net;
    }

    public void setInputs(double[] inputs) {
        for (int i = 0; i < inputs.length; i++) {
            var connection = inputConnections.get(i);
            connection.setInput(inputs[i]);
        }
    }

    public void buildConnections() {
        for (int i = 0; i < numConnections; i++) {
            var connection = addInputConnection();
            addOutputConnection(connection, i);
        }
        addInputConnection(); // bias
    }

    private Connection addInputConnection() {
        var connection = new Connection(networkBuilder);
        connection.setInputNeuron(this);
        inputConnections.add(connection);
        return connection;
    }

    private void addOutputConnection(Connection outputConnection,
                                     int prevNeuronIndex) {
        if (previousLayer != null) {
            HiddenNeuron prevNeuron = previousLayer.getNeurons().get(prevNeuronIndex);
            prevNeuron.addOutputConnection(outputConnection);
        }
    }

    public void adjustWeights() {
        for (Connection connection : inputConnections) {
            connection.adjustWeight();
        }
    }
}
