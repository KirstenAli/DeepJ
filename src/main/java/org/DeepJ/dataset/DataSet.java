package org.DeepJ.dataset;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class DataSet implements Serializable {
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

    public List<Row> getRows() {
        return rows;
    }

    public void setRows(List<Row> rows) {
        this.rows = rows;
    }

    public int getInputDimension() {
        return inputDimension;
    }

    public void setInputDimension(int inputDimension) {
        this.inputDimension = inputDimension;
    }

    public int getOutputDimension() {
        return outputDimension;
    }

    public void setOutputDimension(int outputDimension) {
        this.outputDimension = outputDimension;
    }
}
