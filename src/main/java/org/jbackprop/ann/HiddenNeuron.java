package org.jbackprop.ann;

import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;

@Getter @Setter
public class HiddenNeuron extends Neuron{

    private List<Connection> outputConnections;

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
