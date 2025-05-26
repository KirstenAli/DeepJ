package org.DeepJ.annoop;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class HiddenNeuron extends Neuron{

    private List<Connection> outputConnections;

    public HiddenNeuron(Integer numConnections, HiddenLayer previousLayer, NetworkBuilder networkBuilder) {
        super(numConnections, previousLayer, networkBuilder);
        outputConnections = new ArrayList<>();
    }

    public void calculateDelta() {
        double weightedDeltaSum = sumWeightedDeltasFromOutputs();
        var activationDerivative = activationFunction.derivative(net, activation);
        delta = activationDerivative * weightedDeltaSum;
    }

    public double sumWeightedDeltasFromOutputs() {
        return sumWeightedDeltas(outputConnections);
    }

    public double sumWeightedDeltasFromInputs() {
        return sumWeightedDeltas(inputConnections.subList(0, inputConnections.size() - 1));
    }

    private double sumWeightedDeltas(Collection<Connection> connections) {
        return connections.stream()
                .mapToDouble(Connection::calculateWeightedDelta)
                .sum();
    }

    void addOutputConnection(Connection outputConnection){
        outputConnection.setOutputNeuron(this);
        outputConnections.add(outputConnection);
    }

    public List<Connection> getOutputConnections() {
        return outputConnections;
    }

    public void setOutputConnections(List<Connection> outputConnections) {
        this.outputConnections = outputConnections;
    }
}
