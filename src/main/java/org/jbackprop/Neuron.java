package org.jbackprop;

import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;

public abstract class Neuron {
    private double bias;
    private double net;
    @Getter
    private double activation;

    @Getter
    private double delta;

    private final List<Connection> inputConnections;

    @Setter
    private List<Connection> outputConnections;

    private LossFunction lossFunction;

    public Neuron(int numConnections,
                  Layer previousLayer,
                  LossFunction lossFunction){
        inputConnections = new ArrayList<>();
        this.lossFunction = lossFunction;
        buildConnections(numConnections, previousLayer);
    }

    abstract Double activationFunction(double net);
    abstract Double dActivation(double net);

    public double calculateLoss(double target){
        return lossFunction.calculateLoss(target, activation);
    }

    public double dLoss(double target){
        return lossFunction.dLoss(target, activation);
    }

    public Double calculateActivation(){
        activation = activationFunction(calculateNet());
        return activation;
    }

    public double calculateDelta(){
        double weightedDeltaSum =0;

        for(Connection connection: outputConnections){
            weightedDeltaSum+= connection.calculateWeightedDelta();
        }
        delta = dActivation(net)*weightedDeltaSum;

        return delta;
    }

    public double calculateDelta(double target){
        var dLoss = lossFunction.dLoss(target, activation);
        delta = dActivation(net)*dLoss;

        return delta;
    }

    public double calculateNet(){
        for(Connection connection: inputConnections)
            net+= connection.calculateProduct();

        net+=bias;
        return net;
    }

    public void setInputs(List<Double> inputs){
        for (int i=0; i<inputs.size(); i++){
            var connection = inputConnections.get(i);
            connection.setInput(inputs.get(i));
        }
    }

    private void buildConnections(int numConnections,
                                  Layer previousLayer){
        for (int i=0; i<numConnections; i++){
            var connection = new Connection();
            connection.setOutputNeuron(this);
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
