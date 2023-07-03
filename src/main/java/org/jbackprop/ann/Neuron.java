package org.jbackprop.ann;

import lombok.Getter;
import lombok.Setter;
import org.jbackprop.ann.activationfunctions.ActivationFunction;

import java.util.ArrayList;
import java.util.List;

@Setter
@Getter
public abstract class Neuron {
    protected double net;
    protected double activation;

    protected double delta;

    private final List<Connection> inputConnections;

    private Connection bias;

    protected final ActivationFunction activationFunction;
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
        net = 0; // resets net for next forward pass.
    }

}
