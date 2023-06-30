package org.jbackprop.test;

import org.jbackprop.ann.MSE;
import org.jbackprop.ann.NetworkParams;
import org.jbackprop.ann.SigmoidNeuron;
import org.jbackprop.dataset.DataSet;
import org.jbackprop.dataset.Row;

import java.util.List;

public class Test{
    public static void main(String[] args){
        var params = new NetworkParams<>(SigmoidNeuron.class,
                new MSE(),
                0.1,
                1000000);

        var rows = List.of(
                new Row(List.of(0.0,0.0), List.of(0.0)),
                new Row(List.of(1.0,0.0), List.of(1.0)),
                new Row(List.of(0.0,1.0), List.of(1.0)),
                new Row(List.of(1.0,1.0), List.of(0.0)));

        var dataset = new DataSet(2,1);
        dataset.addRows(rows);

        new MyNetwork(params, dataset, 2,3,1);
    }
}
