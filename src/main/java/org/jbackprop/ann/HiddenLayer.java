package org.jbackprop.ann;

public class HiddenLayer extends Layer<HiddenNeuron>{

    @Override
    void build(int numNeurons, int connectionsPerNeuron, HiddenLayer previousLayer, NetworkBuilder networkBuilder) {
        this.neurons =
                LayerBuilder.build(numNeurons, connectionsPerNeuron, previousLayer, networkBuilder, HiddenNeuron.class);
    }

    public void calculateDeltas(){
        for(HiddenNeuron neuron: neurons)
            neuron.calculateDelta();
    }
}
