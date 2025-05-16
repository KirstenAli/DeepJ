package org.DeepJ.transformer;

import java.util.Random;

public class Tensor {
    public double[][] data;
    public int rows, cols;

    public Tensor(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    public Tensor(double[][] data) {
        this.rows = data.length;
        this.cols = data[0].length;
        this.data = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            System.arraycopy(data[i], 0, this.data[i], 0, cols);
    }

    public static Tensor random(int rows, int cols, Random rand) {
        Tensor t = new Tensor(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                t.data[i][j] = rand.nextGaussian() * 0.1;
        return t;
    }

    public Tensor matmul(Tensor other) {
        Tensor result = new Tensor(this.rows, other.cols);
        for (int i = 0; i < this.rows; i++)
            for (int j = 0; j < other.cols; j++)
                for (int k = 0; k < this.cols; k++)
                    result.data[i][j] += this.data[i][k] * other.data[k][j];
        return result;
    }

    public Tensor transpose() {
        Tensor result = new Tensor(this.cols, this.rows);
        for (int i = 0; i < this.rows; i++)
            for (int j = 0; j < this.cols; j++)
                result.data[j][i] = this.data[i][j];
        return result;
    }

    public Tensor scale(double scalar) {
        Tensor result = new Tensor(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++)
            for (int j = 0; j < this.cols; j++)
                result.data[i][j] = this.data[i][j] * scalar;
        return result;
    }

    public static Tensor softmaxRows(Tensor logits) {
        Tensor result = new Tensor(logits.rows, logits.cols);
        for (int i = 0; i < logits.rows; i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < logits.cols; j++)
                max = Math.max(max, logits.data[i][j]);

            double sum = 0.0;
            for (int j = 0; j < logits.cols; j++)
                sum += Math.exp(logits.data[i][j] - max);

            for (int j = 0; j < logits.cols; j++)
                result.data[i][j] = Math.exp(logits.data[i][j] - max) / sum;
        }
        return result;
    }

    public static Tensor applySoftmaxBackward(Tensor dL_dSoftmax, Tensor softmaxOutput) {
        int rows = dL_dSoftmax.rows;
        int cols = dL_dSoftmax.cols;
        Tensor result = new Tensor(rows, cols);

        for (int i = 0; i < rows; i++) {
            double[] grad = dL_dSoftmax.data[i];
            double[] s = softmaxOutput.data[i];
            for (int j = 0; j < cols; j++) {
                double sum = 0.0;
                for (int k = 0; k < cols; k++) {
                    double jac = (j == k) ? s[j] * (1 - s[j]) : -s[j] * s[k];
                    sum += jac * grad[k];
                }
                result.data[i][j] = sum;
            }
        }

        return result;
    }

    public Tensor subtract(Tensor other) {
        Tensor result = new Tensor(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++)
            for (int j = 0; j < this.cols; j++)
                result.data[i][j] = this.data[i][j] - other.data[i][j];
        return result;
    }

    public double mseLoss(Tensor target) {
        double sum = 0.0;
        for (int i = 0; i < this.rows; i++)
            for (int j = 0; j < this.cols; j++)
                sum += Math.pow(this.data[i][j] - target.data[i][j], 2);
        return sum / (this.rows * this.cols);
    }

    public static Tensor unflattenToTensor(double[] flat, int rows, int cols) {
        Tensor t = new Tensor(rows, cols);
        for (int i = 0; i < flat.length; i++) {
            t.data[i / cols][i % cols] = flat[i];
        }
        return t;
    }

    public static double[] flattenTensor(Tensor t) {
        double[] flat = new double[t.rows * t.cols];
        int index = 0;
        for (int i = 0; i < t.rows; i++) {
            for (int j = 0; j < t.cols; j++) {
                flat[index++] = t.data[i][j];
            }
        }
        return flat;
    }

    public void print(String label) {
        System.out.println(label);
        for (double[] row : data) {
            for (double val : row)
                System.out.printf("%.4f ", val);
            System.out.println();
        }
    }
}
