package org.DeepJ.annv2.layers;

import org.DeepJ.annv2.Tensor;

public class FlattenLayer implements Layer {
    private int originalRows;
    private int originalCols;

    @Override
    public Tensor forward(Tensor input) {
        this.originalRows = input.rows;
        this.originalCols = input.cols;

        double[] flat = Tensor.flattenTensor(input);
        return new Tensor(new double[][]{flat});
    }

    @Override
    public Tensor backward(Tensor gradOutput, double learningRate) {
        double[] flat = gradOutput.data[0];
        return Tensor.unflattenToTensor(flat, originalRows, originalCols);
    }
}
