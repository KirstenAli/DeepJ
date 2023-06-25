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

    public Neuron(int numConnections, Layer previousLayer){
        this.numConnections = numConnections;
        inputConnections = new ArrayList<>();
        buildConnections(previousLayer);
    }

    public double calculateNet(){
        for(Connection connection: inputConnections)
            net+= connection.calculateProduct();

        return net+=bias;
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

    private void buildConnections(Layer previousLayer){
        for (int i=0; i<numConnections; i++){
            var connection = new Connection();
            inputConnections.add(connection);

            addOutputConnection(previousLayer,connection);
        }
    }

    private void addOutputConnection(Layer previousLayer,
                                     Connection outputConnection){
        for(Neuron neuron: previousLayer.getNeurons())
            neuron.addOutputConnection(outputConnection);
    }

    private void addOutputConnection(Connection outputConnection){
        outputConnections.add(outputConnection);
    }
}
