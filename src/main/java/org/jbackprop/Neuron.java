package org.jbackprop;

import java.util.ArrayList;
import java.util.List;

public class Neuron {
    private double bias;
    private float net;
    private float activation;

    private float delta;

    private final List<Connection> connections = new ArrayList<>();

    private final int numConnections;

    public Neuron(int numConnections) {
        this.numConnections = numConnections;
        buildConnections();
    }

    public void calculateNet(){
        for(Connection connection: connections){
            net+= connection.calculateProduct();
        }
    }

    public Double calculateActivation(){
        calculateNet();
        return 0.0;
    }

    public void setInputs(List<Double> inputs){
        for (int i=0; i<inputs.size(); i++){
            var connection = connections.get(i);
            connection.setInput(inputs.get(i));
        }
    }

    private void buildConnections(){
        for (int i=0; i<numConnections; i++){
           connections.add(new Connection());
        }
    }
}
