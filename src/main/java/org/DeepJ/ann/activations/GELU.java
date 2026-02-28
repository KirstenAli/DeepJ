package org.DeepJ.ann.activations;

import org.DeepJ.ann.Tensor;

/**
 * Gaussian Error Linear Unit (GELU), using the tanh approximation popularized by GPT-2.
 *
 * <p>Forward: x * 0.5 * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))).
 */
public final class GELU implements ActivationFunction {

    private Tensor lastX;

    @Override
    public Tensor forward(Tensor input) {
        lastX = input;
        Tensor out = new Tensor(input.rows, input.cols);

        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                double x = input.data[i][j];
                out.data[i][j] = gelu(x);
            }
        }
        return out;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        if (lastX == null) {
            throw new IllegalStateException("GELU.backward called before forward");
        }
        if (gradOutput.rows != lastX.rows || gradOutput.cols != lastX.cols) {
            throw new IllegalArgumentException("gradOutput shape must match input shape");
        }

        Tensor grad = new Tensor(lastX.rows, lastX.cols);
        for (int i = 0; i < lastX.rows; i++) {
            for (int j = 0; j < lastX.cols; j++) {
                double x = lastX.data[i][j];
                grad.data[i][j] = gradOutput.data[i][j] * geluDeriv(x);
            }
        }
        return grad;
    }

    private static double gelu(double x) {
        // tanh approximation: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
        double c = Math.sqrt(2.0 / Math.PI);
        double x3 = x * x * x;
        double t = c * (x + 0.044715 * x3);
        return 0.5 * x * (1.0 + Math.tanh(t));
    }

    private static double geluDeriv(double x) {
        // derivative of the tanh approximation
        double c = Math.sqrt(2.0 / Math.PI);
        double x2 = x * x;
        double x3 = x2 * x;
        double t = c * (x + 0.044715 * x3);

        double tanhT = Math.tanh(t);
        double sech2 = 1.0 - tanhT * tanhT;

        double dt_dx = c * (1.0 + 3.0 * 0.044715 * x2);
        double d_gelu = 0.5 * (1.0 + tanhT) + 0.5 * x * sech2 * dt_dx;
        return d_gelu;
    }
}
