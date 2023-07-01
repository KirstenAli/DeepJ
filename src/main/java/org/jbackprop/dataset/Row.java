package org.jbackprop.dataset;

import lombok.Getter;
import lombok.Setter;

@Getter @Setter
public class Row{
    private double[] inputs;
    private double[] targets;

    public Row(double[] inputs, double[] targets){
        this.inputs = inputs;
        this.targets = targets;
    }
}
