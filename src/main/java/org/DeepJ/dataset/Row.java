package org.DeepJ.dataset;

import lombok.Getter;
import lombok.Setter;

import java.io.Serializable;

@Getter @Setter
public class Row implements Serializable {
    private double[] inputs;
    private double[] targets;

    public Row(double[] inputs, double[] targets){
        this.inputs = inputs;
        this.targets = targets;
    }
}
