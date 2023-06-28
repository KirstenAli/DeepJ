package org.jbackprop;

import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;
@Getter @Setter
public abstract class Neuron {
    private double net;
    private double activation;

    private double delta;

    private final List<Connection> inputConnections;

    private List<Connection> outputConnections;

    private final LossFunction lossFunction;

    private double loss;
    private Connection bias;

    public Neuron(int numConnections,
                  Layer previousLayer,
                  GlobalParams globalParams){

        inputConnections = new ArrayList<>();
        this.lossFunction = globalParams.getLossFunction();

        buildConnections(numConnections, previousLayer, globalParams);
    }

    abstract double activationFunction(double net);
    abstract double dActivation(double net);

    public double calculateLoss(double target){
        loss = lossFunction.calculateLoss(target, activation);
        return loss;
    }

    public double dLoss(double target){
        return lossFunction.dLoss(target, activation);
    }

    public double calculateActivation(){
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
        var dLoss = dLoss(target);
        delta = dActivation(net)*dLoss;

        return delta;
    }

    public double calculateNet(){
        for(Connection connection: inputConnections)
            net+= connection.calculateProduct();

        net+=bias.getProduct();
        return net;
    }

    public void setInputs(List<Double> inputs){
        for (int i=0; i<inputs.size(); i++){
            var connection = inputConnections.get(i);
            connection.setInput(inputs.get(i));
        }
    }

    private void buildConnections(int numConnections,
                                  Layer previousLayer,
                                  GlobalParams globalParams){
        bias = new Connection(globalParams);
        inputConnections.add(bias);

        for (int i=0; i<numConnections; i++){
            var connection = new Connection(globalParams);
            connection.setInputNeuron(this);
            inputConnections.add(connection);

            addOutputConnection(previousLayer,connection,i);
        }
    }

    private void addOutputConnection(Layer previousLayer,
                                     Connection outputConnection,
                                     int index){
        if(previousLayer!=null){
            var previousNeuron = previousLayer.getNeurons().get(index);
            outputConnection.setOutputNeuron(previousNeuron);
        }
    }


}
