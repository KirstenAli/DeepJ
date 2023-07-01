package org.jbackprop.ann;

public class OutputLayer extends Layer{

    public void calculateDeltas(double[] targets){
        for(int i=0; i<targets.length; i++)
            neurons.get(i).calculateDelta(targets[i]);
    }
}
