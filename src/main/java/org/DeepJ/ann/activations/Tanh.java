package org.DeepJ.ann.activations;

import org.DeepJ.transformer.Tensor;

public class Tanh implements ActivationFunction {
    private Tensor output;            // cache

    @Override
    public Tensor forward(Tensor input) {
        output = new Tensor(input.rows, input.cols);
        Tensor.matrixOp((r, c) -> {
            output.data[r][c] = Math.tanh(input.data[r][c]);
        }, input);
        return output;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        Tensor grad = new Tensor(output.rows, output.cols);
        Tensor.matrixOp((r, c) -> {
            double y = output.data[r][c];
            grad.data[r][c] = gradOutput.data[r][c] * (1.0 - y * y);
        }, output);
        return grad;
    }
}