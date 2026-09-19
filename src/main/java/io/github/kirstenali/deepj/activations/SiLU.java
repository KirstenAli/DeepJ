package io.github.kirstenali.deepj.activations;

import io.github.kirstenali.deepj.tensor.Tensor;

public final class SiLU implements ActivationFunction {

    private Tensor lastX;
    private Tensor lastSigmoid;

    @Override
    public Tensor forward(Tensor input) {
        lastX = input;
        lastSigmoid = input.sigmoidActivation();
        return input.multiply(lastSigmoid);
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        if (lastX == null) {
            throw new IllegalStateException("SiLU.backward called before forward");
        }
        if (gradOutput.rows != lastX.rows || gradOutput.cols != lastX.cols) {
            throw new IllegalArgumentException("gradOutput shape must match input shape");
        }

        Tensor oneMinusSig = lastSigmoid.multiplyScalar(-1.0f);
        oneMinusSig.addScalarInPlace(1.0f);

        Tensor xSig = lastX.multiply(lastSigmoid);
        xSig.multiplyInPlace(oneMinusSig);

        Tensor dSiLU = lastSigmoid.add(xSig);
        return gradOutput.multiply(dSiLU);
    }
}
