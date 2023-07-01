package org.jbackprop.ann;

public class OutputNeuron extends Neuron{
    public OutputNeuron(Integer numConnections, Layer previousLayer, NetworkParams networkParams) {
        super(numConnections, previousLayer, networkParams);
    }
}
