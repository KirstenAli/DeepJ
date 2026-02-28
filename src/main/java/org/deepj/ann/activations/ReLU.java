package org.deepj.ann.activations;

import org.deepj.ann.Tensor;

public class ReLU implements ActivationFunction {
    private Tensor input;

    @Override
    public Tensor forward(Tensor input) {
        this.input = input;
        Tensor result = new Tensor(input.rows, input.cols);
        for (int r = 0; r < input.rows; r++)
            for (int c = 0; c < input.cols; c++)
                result.data[r][c] = Math.max(0, input.data[r][c]);
        return result;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        Tensor grad = new Tensor(input.rows, input.cols);
        for (int r = 0; r < input.rows; r++)
            for (int c = 0; c < input.cols; c++)
                grad.data[r][c] = input.data[r][c] > 0 ? gradOutput.data[r][c] : 0;
        return grad;
    }
}
