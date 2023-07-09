package org.jbackprop.ann;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.Getter;
import lombok.Setter;
import org.jbackprop.ann.activationfunctions.ActivationFunction;

import java.util.ArrayList;
import java.util.List;

@Setter
@Getter
public abstract class Neuron {
    @JsonIgnore
    protected double net;
    @JsonIgnore
    protected double activation;
    @JsonIgnore
    protected double delta;
    private List<Connection> inputConnections;
    private Connection bias;
    @JsonIgnore
    protected ActivationFunction activationFunction;
    @JsonIgnore
    private int numConnections;
    @JsonIgnore
    private HiddenLayer previousLayer;
    @JsonIgnore
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

        net += bias.getProduct();
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
            var connection = new Connection(networkBuilder);
            connection.setInputNeuron(this);
            inputConnections.add(connection);

            addOutputConnection(connection, i);
        }

        bias = new Connection(networkBuilder);
        bias.setInputNeuron(this);
        inputConnections.add(bias);
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
