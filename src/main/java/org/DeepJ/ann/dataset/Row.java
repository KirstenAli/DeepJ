package org.DeepJ.ann.dataset;

import java.io.Serializable;

public class Row implements Serializable {
    private double[] inputs;
    private double[] targets;

    public Row(double[] inputs, double[] targets){
        this.inputs = inputs;
        this.targets = targets;
    }

    public double[] getInputs() {
        return inputs;
    }

    public void setInputs(double[] inputs) {
        this.inputs = inputs;
    }

    public double[] getTargets() {
        return targets;
    }

    public void setTargets(double[] targets) {
        this.targets = targets;
    }
}
