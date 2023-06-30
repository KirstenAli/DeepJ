package org.jbackprop.dataset;

import lombok.Getter;
import lombok.Setter;

import java.util.List;
@Getter @Setter
public class Row{
    private List<Double> input;
    private List<Double> target;

    public Row(List<Double> input, List<Double> target) {
        this.input = input;
        this.target = target;
    }
}
