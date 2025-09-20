package org.DeepJ.ann.activations;

import org.DeepJ.ann.Tensor;

public class Tanh implements ActivationFunction {
    private Tensor output;

    @Override
    public Tensor forward(Tensor input) {
        output = new Tensor(input.rows, input.cols);
        output.iterate((r, c) -> output.data[r][c] = Math.tanh(input.data[r][c]));
        return output;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        Tensor grad = new Tensor(output.rows, output.cols);
        grad.iterate((r, c) -> {
            double y = output.data[r][c];
            grad.data[r][c] = gradOutput.data[r][c] * (1.0 - y * y);
        });
        return grad;
    }
}