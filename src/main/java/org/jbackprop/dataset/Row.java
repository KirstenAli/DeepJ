package org.jbackprop.dataset;

import lombok.Getter;
import lombok.Setter;

@Getter @Setter
public class Row{
    private double[] input;
    private double[] target;

    public Row(double[] input, double[] target){
        this.input = input;
        this.target = target;
    }
}
