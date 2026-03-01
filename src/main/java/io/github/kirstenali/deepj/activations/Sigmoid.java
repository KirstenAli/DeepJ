package io.github.kirstenali.deepj.activations;

import io.github.kirstenali.deepj.tensor.Tensor;

public class Sigmoid implements ActivationFunction {
    private Tensor output;

    @Override
    public Tensor forward(Tensor input) {
        output = new Tensor(input.rows, input.cols);
        for (int r = 0; r < input.rows; r++)
            for (int c = 0; c < input.cols; c++)
                output.data[r][c] = 1.0 / (1.0 + Math.exp(-input.data[r][c]));
        return output;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        Tensor grad = new Tensor(output.rows, output.cols);
        for (int r = 0; r < output.rows; r++) {
            for (int c = 0; c < output.cols; c++) {
                double sig = output.data[r][c];
                grad.data[r][c] = gradOutput.data[r][c] * sig * (1.0 - sig);
            }
        }
        return grad;
    }
}