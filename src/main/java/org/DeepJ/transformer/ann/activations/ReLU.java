package org.DeepJ.transformer.ann.activations;

import org.DeepJ.transformer.Tensor;

public class ReLU implements ActivationFunction {
    private Tensor input;

    @Override
    public Tensor forward(Tensor input) {
        this.input = input;
        Tensor result = new Tensor(input.rows, input.cols);
        Tensor.matrixOp((r,c) -> result.data[r][c] = Math.max(0, input.data[r][c]), input);
        return result;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        Tensor grad = new Tensor(input.rows, input.cols);
        Tensor.matrixOp((r,c) -> grad.data[r][c] = input.data[r][c] > 0 ? gradOutput.data[r][c] : 0, input);
        return grad;
    }
}
