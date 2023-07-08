package org.jbackprop.dataset;

import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;
@Setter @Getter
public class DataSet {
    private List<Row> rows;
    private int inputDimension;
    private int outputDimension;

    public DataSet(int inputDimension, int outputDimension) {
        rows = new ArrayList<>();
        this.inputDimension = inputDimension;
        this.outputDimension = outputDimension;
    }

    public void addRow(Row row){
        rows.add(row);
    }

    public void addRow(double[] input, double[]output){
        rows.add(new Row(input, output));
    }
}
