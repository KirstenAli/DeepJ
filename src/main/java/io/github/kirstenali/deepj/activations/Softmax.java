package io.github.kirstenali.deepj.activations;

import io.github.kirstenali.deepj.tensor.Tensor;

/**
 * Row-wise softmax for 2D tensors: applies softmax independently to each row.
 *
 * Forward:  softmaxRows(logits)
 * Backward: given upstream grad dY, returns dLogits using cached softmax output.
 */
public final class Softmax implements ActivationFunction {

    private Tensor softmaxOut; // cache from forward

    @Override
    public Tensor forward(Tensor logits) {
        Tensor result = new Tensor(logits.rows, logits.cols);
        for (int i = 0; i < logits.rows; i++) {
            computeSoftmaxRow(logits.data[i], result.data[i]);
        }
        this.softmaxOut = result;
        return result;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        if (softmaxOut == null) {
            throw new IllegalStateException("SoftmaxRows.backward() called before forward()");
        }

        Tensor.requireSameShape(gradOutput, softmaxOut, "SoftmaxRows.backward");

        Tensor gradWrtLogits = new Tensor(gradOutput.rows, gradOutput.cols);

        for (int row = 0; row < gradOutput.rows; row++) {
            double[] gradRow = gradOutput.data[row];
            double[] probRow = softmaxOut.data[row];
            double[] outRow = gradWrtLogits.data[row];

            double weightedSum = dotProduct(gradRow, probRow);

            for (int j = 0; j < gradRow.length; j++) {
                outRow[j] = probRow[j] * (gradRow[j] - weightedSum);
            }
        }
        return gradWrtLogits;
    }

    public static void computeSoftmaxRow(double[] inputRow, double[] outputRow) {
        double max = findMax(inputRow);
        double sum = computeExpSum(inputRow, max);
        for (int j = 0; j < inputRow.length; j++) {
            outputRow[j] = Math.exp(inputRow[j] - max) / sum;
        }
    }

    public static double findMax(double[] row) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : row) {
            if (v > max) {
                max = v;
            }
        }
        return max;
    }

    public static double computeExpSum(double[] row, double max) {
        double sum = 0.0;
        for (double v : row) {
            sum += Math.exp(v - max);
        }
        return sum;
    }

    private static double dotProduct(double[] a, double[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }
}