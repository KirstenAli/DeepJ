package org.jbackprop.ann;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.Getter;

@Getter
public class OutputLayer extends Layer<OutputNeuron> {
    private int size;
    public void calculateDeltas(double[] targets) {
        for (int i = 0; i < targets.length; i++)
            neurons.get(i).calculateDelta(targets[i]);
    }

    @Override
    void build(int numNeurons, int connectionsPerNeuron, HiddenLayer previousLayer, NetworkBuilder networkBuilder) {
        size = numNeurons;
        this.neurons =
                LayerBuilder.build(numNeurons, connectionsPerNeuron, previousLayer, networkBuilder, OutputNeuron.class);
    }
}
