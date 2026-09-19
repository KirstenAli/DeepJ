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

        Tensor tanhSq = output.multiply(output);
        tanhSq.multiplyScalarInPlace(-1.0f);
        tanhSq.addScalarInPlace(1.0f);

        return gradOutput.multiply(tanhSq);
    }
}
