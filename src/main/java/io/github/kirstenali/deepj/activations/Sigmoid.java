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
        Tensor oneMinusSig = output.multiplyScalar(-1.0f);
        oneMinusSig.addScalarInPlace(1.0f);

        Tensor grad = gradOutput.multiply(output);
        grad.multiplyInPlace(oneMinusSig);
        return grad;
    }
}