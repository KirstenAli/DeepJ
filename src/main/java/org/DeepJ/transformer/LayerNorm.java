package org.DeepJ.transformer;

public class LayerNorm {
    private final int dim;
    private static final double epsilon = 1e-5;

    private Tensor gamma;
    private Tensor beta;
    private double learningRate = 1e-3;

    private Tensor input, mean, variance, normalized, dL_dOutput;

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
        return x.subtractRows(mean)
                .divideRows(var.addScalar(epsilon).sqrt());
    }

    private Tensor applyAffine(Tensor norm) {
        return norm.multiplyBroadcast(gamma).addBroadcast(beta);
    }

    public Tensor backward(Tensor dL_dOutput) {
        this.dL_dOutput = dL_dOutput;

        Tensor std  = variance.addScalar(epsilon).sqrt();
        Tensor xMu  = input.subtractRows(mean);
        Tensor dNorm = dL_dOutput.multiplyBroadcast(gamma);

        Tensor dVar  = computeDVariance(dNorm, xMu, std);
        Tensor dMean = computeDMean(dNorm, xMu, std, dVar);

        return computeDInput(dNorm, xMu, std, dMean, dVar);
    }

    private Tensor computeDVariance(Tensor dNorm, Tensor xMu, Tensor std) {
        return dNorm.multiply(xMu)
                .multiplyScalar(-0.5)
                .multiplyRows(std.pow(-3))
                .sumAlongRows();
    }

    private Tensor computeDMean(Tensor dNorm, Tensor xMu, Tensor std, Tensor dVar) {
        return dNorm.divideRows(std)
                .multiplyScalar(-1.0)
                .sumAlongRows()
                .add(xMu.multiplyScalar(-2.0)
                        .multiplyRows(dVar)
                        .sumAlongRows()
                        .divideScalar(input.cols)
                );
    }

    private Tensor computeDInput(Tensor dNorm, Tensor xMu, Tensor std, Tensor dMean, Tensor dVar) {
        return dNorm.divideRows(std)
                .add(xMu.multiplyScalar(2.0)
                        .multiplyRows(dVar)
                        .divideScalar(input.cols)
                )
                .addRows(dMean.divideScalar(input.cols));
    }

    public void updateWeights() {
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
