package io.github.kirstenali.deepj.activations;

import io.github.kirstenali.deepj.tensor.Tensor;

public class Sigmoid implements ActivationFunction {
    private Tensor output;

    @Override
    public Tensor forward(Tensor input) {
        output = input.sigmoidActivation();
        return output;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        // d_sigmoid = sigmoid * (1 - sigmoid) * grad
        Tensor ones = Tensor.ones(output.rows, output.cols);
        Tensor oneMinusSig = ones.subtract(output);
        return gradOutput.multiply(output).multiply(oneMinusSig);
    }
}