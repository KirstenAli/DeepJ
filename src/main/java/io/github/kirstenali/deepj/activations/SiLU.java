package io.github.kirstenali.deepj.activations;

import io.github.kirstenali.deepj.tensor.Tensor;

/**
 * Sigmoid Linear Unit (SiLU / Swish).
 *
 * <p>Forward:  SiLU(x) = x · σ(x)
 * <p>Backward: SiLU'(x) = σ(x) · (1 + x · (1 − σ(x)))
 *
 * <p>Used as the gate activation inside {@link io.github.kirstenali.deepj.layers.transformer.SwiGLULayer}
 * and is the standard FFN activation in Llama, Mistral, and Qwen.
 */
public final class SiLU implements ActivationFunction {

    private Tensor lastX;
    private Tensor lastSigmoid;

    @Override
    public Tensor forward(Tensor input) {
        lastX = input;
        lastSigmoid = input.sigmoidActivation();   // σ(x)
        return input.multiply(lastSigmoid);         // x · σ(x)
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        if (lastX == null) {
            throw new IllegalStateException("SiLU.backward called before forward");
        }
        if (gradOutput.rows != lastX.rows || gradOutput.cols != lastX.cols) {
            throw new IllegalArgumentException("gradOutput shape must match input shape");
        }

        // SiLU'(x) = σ(x) · (1 + x · (1 − σ(x)))
        //           = σ(x) + x · σ(x) · (1 − σ(x))
        Tensor oneMinusSig = lastSigmoid.multiplyScalar(-1.0); // 1 - σ(x)
        oneMinusSig.addScalarInPlace(1.0);

        Tensor xSig = lastX.multiply(lastSigmoid);
        xSig.multiplyInPlace(oneMinusSig);

        Tensor dSiLU = lastSigmoid.add(xSig);
        return gradOutput.multiply(dSiLU);
    }
}

