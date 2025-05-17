package org.DeepJ.ann;

public class OutputLayer extends Layer<OutputNeuron> {
    public void calculateDeltas(double[] targets) {
        for (int i = 0; i < targets.length; i++)
            neurons.get(i).calculateDelta(targets[i]);
    }

    @Override
    void build(int numNeurons, int connectionsPerNeuron, HiddenLayer previousLayer, NetworkBuilder networkBuilder) {
        this.neurons =
                LayerBuilder.build(numNeurons, connectionsPerNeuron, previousLayer, networkBuilder, OutputNeuron.class);
    }

    public int getSize() {
        return neurons.size();
    }
}
