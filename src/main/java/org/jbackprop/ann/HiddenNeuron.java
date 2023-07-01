package org.jbackprop.ann;

import java.util.ArrayList;
import java.util.List;

public class HiddenNeuron extends Neuron{

    private final List<Connection> outputConnections;
    public HiddenNeuron(Integer numConnections, HiddenLayer previousLayer, NetworkBuilder networkBuilder) {
        super(numConnections, previousLayer, networkBuilder);
        outputConnections = new ArrayList<>();
    }

    public void calculateDelta(){
        double weightedDeltaSum =0;

        for(Connection connection: outputConnections){
            weightedDeltaSum+= connection.calculateWeightedDelta();
        }

        var activationDerivative = activationFunction.derivative(net,activation);
        delta = activationDerivative*weightedDeltaSum;
    }

    void addOutputConnection(Connection outputConnection){
        outputConnection.setOutputNeuron(this);
        outputConnections.add(outputConnection);
    }
}
