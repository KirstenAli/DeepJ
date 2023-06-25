package org.jbackprop;

import lombok.Setter;

import java.util.ArrayList;
import java.util.List;

public class Neuron {
    private double bias;
    private float net;
    private float activation;

    private float delta;

    private final List<Connection> inputConnections;

    @Setter
    private List<Connection> outputConnections;

    private final int numConnections;

    public Neuron(int numConnections, List<Neuron> previousNeurons){
        this.numConnections = numConnections;
        inputConnections = new ArrayList<>();
        buildConnections(previousNeurons);
    }

    public void calculateNet(){
        for(Connection connection: inputConnections){
            net+= connection.calculateProduct();
        }
    }

    public Double calculateActivation(){
        calculateNet();
        return 0.0;
    }

    public void setInputs(List<Double> inputs){
        for (int i=0; i<inputs.size(); i++){
            var connection = inputConnections.get(i);
            connection.setInput(inputs.get(i));
        }
    }

    private void buildConnections(List<Neuron> previousNeurons){
        for (int i=0; i<numConnections; i++){
            var connection = new Connection();
            inputConnections.add(connection);

            addOutputConnection(previousNeurons,connection);
        }
    }

    private void addOutputConnection(List<Neuron> previousNeurons,
                                     Connection outputConnection){
        for(Neuron neuron: previousNeurons)
            neuron.addOutputConnection(outputConnection);
    }

    private void addOutputConnection(Connection outputConnection){
        outputConnections.add(outputConnection);
    }
}
