package org.DeepJ.transformer;

public class LayerNorm {
    private final int dim;
    private final double epsilon = 1e-5;
    private Tensor gamma;
    private Tensor beta;

    // Cached during forward pass for backprop
    private Tensor input, mean, variance, normalized;

    public LayerNorm(int dim) {
        this.dim = dim;
        this.gamma = Tensor.ones(1, dim);
        this.beta = Tensor.zeros(1, dim);
    }

    public Tensor forward(Tensor input) {
        this.input = input;

        mean = input.meanAlongRows();           // shape: (rows, 1)
        variance = input.varianceAlongRows();   // shape: (rows, 1)

        normalized = input.subtractRows(mean)
                .divideRows(variance.addScalar(epsilon).sqrt());

        return normalized.multiplyBroadcast(gamma).addBroadcast(beta);
    }

    public Tensor backward(Tensor dL_dOutput, double learningRate) {
        int rows = input.rows;
        int cols = input.cols;

        Tensor std = variance.addScalar(epsilon).sqrt();

        Tensor dNorm = dL_dOutput.multiplyBroadcast(gamma); // dL/dNorm
        Tensor xMu = input.subtractRows(mean);

        // dL/dVariance
        Tensor dVar = dNorm.multiply(xMu).multiplyScalar(-0.5).multiply(std.pow(-3)).sumAlongRows();

        // dL/dMean
        Tensor dMean = dNorm.divideRows(std).scale(-1.0).sumAlongRows()
                .add(dVar.multiply(xMu).scale(-2.0).sumAlongRows().divideScalar(cols));

        // dL/dInput
        Tensor dInput = dNorm.divideRows(std)
                .add(xMu.scale(2.0).multiplyBroadcast(dVar).divideScalar(cols))
                .add(dMean.divideScalar(cols));

        // Update gamma and beta
        Tensor dGamma = dL_dOutput.multiply(normalized).sumAlongRows();
        Tensor dBeta = dL_dOutput.sumAlongRows();

        gamma = gamma.subtract(dGamma.scale(learningRate));
        beta = beta.subtract(dBeta.scale(learningRate));

        return dInput;
    }
}
