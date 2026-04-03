package io.github.kirstenali.deepj.activations;

import io.github.kirstenali.deepj.tensor.Tensor;

public class Tanh implements ActivationFunction {
    private Tensor output;

    @Override
    public Tensor forward(Tensor input) {
        output = input.tanhActivation();
        return output;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        // d_tanh = (1 - tanh^2) * grad
        Tensor ones = Tensor.ones(output.rows, output.cols);
        Tensor tanhSq = output.multiply(output);
        return gradOutput.multiply(ones.subtract(tanhSq));
    }
}