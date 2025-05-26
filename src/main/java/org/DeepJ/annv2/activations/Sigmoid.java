package org.DeepJ.annv2.activations;

import org.DeepJ.transformer.Tensor;

public class Sigmoid implements ActivationFunction {
    private Tensor output;

    @Override
    public Tensor forward(Tensor input) {
        output = new Tensor(input.rows, input.cols);
        Tensor.matrixOp((r, c) -> {
            double z   = input.data[r][c];
            output.data[r][c] = 1.0 / (1.0 + Math.exp(-z));
        }, input);
        return output;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        Tensor grad = new Tensor(output.rows, output.cols);
        Tensor.matrixOp((r, c) -> {
            double sig = output.data[r][c];
            grad.data[r][c] = gradOutput.data[r][c] * sig * (1.0 - sig);
        }, output);
        return grad;
    }
}