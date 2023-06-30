package org.jbackprop.dataset;

import lombok.Getter;
import lombok.Setter;

import java.util.List;
@Setter @Getter
public class DataSet {
    List<Row> rows;
    private int inputDimension;
    private int outputDimension;

    public DataSet(int inputDimension, int outputDimension) {
        this.inputDimension = inputDimension;
        this.outputDimension = outputDimension;
    }

    public void addRows(List<Row> rows) {
        this.rows = rows;
    }

    public void addRow(Row row){
        rows.add(row);
    }

    public void addRow(List<Double> input, List<Double> output){
        rows.add(new Row(input, output));
    }
}
