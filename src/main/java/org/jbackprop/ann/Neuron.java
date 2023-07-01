package org.jbackprop.ann;

import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;
@Getter @Setter
public class Neuron {
    private double net;
    private double activation;

    private double delta;

    private final List<Connection> inputConnections;
    private final List<Connection> outputConnections;
    private Connection bias;

    private final LossFunction lossFunction;
    private final ActivationFunction activationFunction;

    private double loss;
    private double actualLoss;

    public Neuron(Integer numConnections,
                  Layer previousLayer,
                  NetworkParams networkParams){

        inputConnections = new ArrayList<>();
        outputConnections = new ArrayList<>();
        this.lossFunction = networkParams.getLossFunction();
        this.activationFunction = networkParams.getActivationFunction();

        buildConnections(numConnections, previousLayer, networkParams);
    }

    private double calculateActualLoss(double target){
        actualLoss = target-activation;
        return actualLoss;
    }

    public double calculateLoss(){
        loss = lossFunction.calculateLoss(actualLoss);
        return loss;
    }

    private double dLoss(double target){
        return lossFunction.dLoss(calculateActualLoss(target));
    }

    public double calculateActivation(){
        var net = calculateNet();
        activation = activationFunction.applyActivation(net);
        return activation;
    }

    public void calculateDelta(){
        double weightedDeltaSum =0;

        for(Connection connection: outputConnections){
            weightedDeltaSum+= connection.calculateWeightedDelta();
        }

        var activationDerivative = activationFunction.derivative(net,activation);
        delta = activationDerivative*weightedDeltaSum;
    }

    public void calculateDelta(double target){
        var dLoss = dLoss(target);
        var activationDerivative = activationFunction.derivative(net,activation);
        delta = activationDerivative*dLoss;
    }

    public double calculateNet(){
        for(Connection connection: inputConnections)
            net+= connection.calculateProduct();

        net+=bias.getProduct();
        return net;
    }

    public void setInputs(double[] inputs){
        for (int i=0; i<inputs.length; i++){
            var connection = inputConnections.get(i);
            connection.setInput(inputs[i]);
        }
    }

    private void buildConnections(int numConnections,
                                  Layer previousLayer,
                                  NetworkParams networkParams){

        for (int i=0; i<numConnections; i++){
            var connection = new Connection(networkParams);
            connection.setInputNeuron(this);
            inputConnections.add(connection);

            addOutputConnection(previousLayer,connection,i);
        }

        bias = new Connection(networkParams);
        bias.setInputNeuron(this);
        inputConnections.add(bias);
    }

    private void addOutputConnection(Layer previousLayer,
                                     Connection outputConnection,
                                     int prevNeuronIndex){
        if(previousLayer!=null){
            var prevNeuron = previousLayer.getNeurons().get(prevNeuronIndex);
            prevNeuron.addOutputConnection(outputConnection);
        }
    }

    private void addOutputConnection(Connection outputConnection){
        outputConnection.setOutputNeuron(this);
        outputConnections.add(outputConnection);
    }

    public void adjustWeights(){
        for(Connection connection: inputConnections){
            connection.adjustWeight();
        }
        net=0; // resets net for next forward pass.
    }
}
