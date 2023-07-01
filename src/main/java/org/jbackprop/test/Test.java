package org.jbackprop.test;

import org.jbackprop.dataset.DataSet;
import org.jbackprop.dataset.Row;

import java.util.List;

public class Test{
    public static void main(String[] args){

        var rows = List.of(
                new Row(new double[]{0.0,0.0}, new double[]{0.0}),
                new Row(new double[]{1.0,0.0}, new double[]{1.0}),
                new Row(new double[]{0.0,1.0}, new double[]{1.0}),
                new Row(new double[]{1.0,1.0}, new double[]{0.0}));

        var dataset = new DataSet(2,1);
        dataset.addRows(rows);

        new MyNetwork(dataset, 3,2,1);
    }
}
