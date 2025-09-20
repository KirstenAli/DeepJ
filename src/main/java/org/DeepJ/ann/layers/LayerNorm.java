package org.DeepJ.ann.layers;

import org.DeepJ.ann.Tensor;

public class LayerNorm implements Layer {
    private static final double epsilon = 1e-5;
    private double learningRate = 1e-3;

    private final int dim;
    private Tensor gamma, beta, input, mean, variance, normalized, dL_dOutput;

    public LayerNorm(int dim) {
        this.dim = dim;
        this.gamma = Tensor.ones(1, dim);
        this.beta  = Tensor.zeros(1, dim);
    }

    public Tensor forward(Tensor input) {
        this.input = input;
        mean = input.meanAlongRows();
        variance = input.varianceAlongRows();
        normalized = normalize(input, mean, variance);

        return applyAffine(normalized);
    }

    private Tensor normalize(Tensor x, Tensor mean, Tensor var) {
        return x.subtractBroadcastCols(mean)
                .divideBroadcastCols(var.addScalar(epsilon).sqrt());
    }

    private Tensor applyAffine(Tensor norm) {
        return norm.multiplyBroadcastRows(gamma).addBroadcastRows(beta);
    }

    public Tensor backward(Tensor dL_dOutput, double learningRate) {
        this.learningRate = learningRate;
        this.dL_dOutput = dL_dOutput;

        Tensor std  = variance.addScalar(epsilon).sqrt();
        Tensor xMu  = input.subtractBroadcastCols(mean);
        Tensor dNorm = dL_dOutput.multiplyBroadcastRows(gamma);

        Tensor dVar  = computeDVariance(dNorm, xMu, std);
        Tensor dMean = computeDMean(dNorm, xMu, std, dVar);

        return computeDInput(dNorm, xMu, std, dMean, dVar);
    }

    private Tensor computeDVariance(Tensor dNorm, Tensor xMu, Tensor std) {
        return dNorm.multiply(xMu)
                .multiplyScalar(-0.5)
                .multiplyBroadcastCols(std.pow(-3))
                .sumAlongRows();
    }

    private Tensor computeDMean(Tensor dNorm, Tensor xMu, Tensor std, Tensor dVar) {
        return dNorm.divideBroadcastCols(std)
                .multiplyScalar(-1.0)
                .sumAlongRows()
                .add(xMu.multiplyScalar(-2.0)
                        .multiplyBroadcastCols(dVar)
                        .sumAlongRows()
                        .divideScalar(input.cols)
                );
    }

    private Tensor computeDInput(Tensor dNorm, Tensor xMu, Tensor std, Tensor dMean, Tensor dVar) {
        return dNorm.divideBroadcastCols(std)
                .add(xMu.multiplyScalar(2.0)
                        .multiplyBroadcastCols(dVar)
                        .divideScalar(input.cols)
                )
                .addBroadcastCols(dMean.divideScalar(input.cols));
    }

    public void step() {
        Tensor dGamma = dL_dOutput.multiply(normalized).sumAlongCols();
        Tensor dBeta  = dL_dOutput.sumAlongCols();

        gamma = gamma.subtract(dGamma.multiplyScalar(learningRate));
        beta  = beta.subtract(dBeta .multiplyScalar(learningRate));
    }

    public int getDim() {
        return dim;
    }

    public double getEpsilon() {
        return epsilon;
    }

    public Tensor getGamma() {
        return gamma;
    }

    public void setGamma(Tensor gamma) {
        this.gamma = gamma;
    }

    public Tensor getBeta() {
        return beta;
    }

    public void setBeta(Tensor beta) {
        this.beta = beta;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public Tensor getInput() {
        return input;
    }

    public void setInput(Tensor input) {
        this.input = input;
    }

    public Tensor getMean() {
        return mean;
    }

    public void setMean(Tensor mean) {
        this.mean = mean;
    }

    public Tensor getVariance() {
        return variance;
    }

    public void setVariance(Tensor variance) {
        this.variance = variance;
    }

    public Tensor getNormalized() {
        return normalized;
    }

    public void setNormalized(Tensor normalized) {
        this.normalized = normalized;
    }

    public Tensor getdL_dOutput() {
        return dL_dOutput;
    }

    public void setdL_dOutput(Tensor dL_dOutput) {
        this.dL_dOutput = dL_dOutput;
    }
}
