package org.DeepJ.annv2;

import java.util.Random;
import java.util.function.BiConsumer;

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

    public void iterate(BiConsumer<Integer, Integer> operation) {
        iterate(operation, this);
    }

    public static void iterate(BiConsumer<Integer, Integer> operation, Tensor t) {
        iterate(operation, t, t);
    }

    public static void iterate(BiConsumer<Integer, Integer> operation, Tensor t, Tensor u) {
        for (int r = 0; r < t.rows; r++)
            for (int c = 0; c < u.cols; c++)
                operation.accept(r,c);
    }

    public static Tensor random(int rows, int cols, Random rand) {
        Tensor t = new Tensor(rows, cols);
        iterate((r, c) -> t.data[r][c] = rand.nextGaussian() * 0.1, t);
        return t;
    }

    public Tensor matmul(Tensor other) {
        Tensor result = new Tensor(this.rows, other.cols);

        iterate((r, c) -> {
            for (int k = 0; k < this.cols; k++)
                result.data[r][c] += this.data[r][k] * other.data[k][c];
        }, this, other);

        return result;
    }

    public Tensor multiply(Tensor other) {
        Tensor result = new Tensor(this.rows, this.cols);
        iterate((r, c) -> result.data[r][c] = this.data[r][c] * other.data[r][c]);
        return result;
    }

    public Tensor multiplyBroadcastCols(Tensor colVector) {
        Tensor result = new Tensor(this.rows, this.cols);
        iterate((r, c) -> result.data[r][c] = this.data[r][c] * colVector.data[r][0]);
        return result;
    }

    public double sum() {
        double[] sum = new double[]{0.0};
        iterate((r, c) -> sum[0] += data[r][c]);
        return sum[0];
    }

    public Tensor transpose() {
        Tensor result = new Tensor(this.cols, this.rows);
        iterate((r, c)-> result.data[c][r] = this.data[r][c]);
        return result;
    }

    public Tensor multiplyScalar(double scalar) {
        Tensor result = new Tensor(this.rows, this.cols);
        iterate((r, c) -> result.data[r][c] = this.data[r][c] * scalar);
        return result;
    }

    public Tensor addScalar(double scalar) {
        Tensor result = new Tensor(this.rows, this.cols);
        iterate((r, c) -> result.data[r][c] = this.data[r][c] + scalar);
        return result;
    }

    public Tensor divideScalar(double scalar) {
        Tensor result = new Tensor(this.rows, this.cols);
        iterate((r, c) -> result.data[r][c] = this.data[r][c] / scalar);
        return result;
    }

    public static Tensor softmaxRows(Tensor logits) {
        Tensor result = new Tensor(logits.rows, logits.cols);
        for (int i = 0; i < logits.rows; i++) {
            computeSoftmaxRow(logits.data[i], result.data[i]);
        }
        return result;
    }

    private static void computeSoftmaxRow(double[] inputRow, double[] outputRow) {
        double max = findMax(inputRow);
        double sum = computeExpSum(inputRow, max);
        applySoftmax(inputRow, outputRow, max, sum);
    }

    private static double findMax(double[] row) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : row) {
            if (v > max) max = v;
        }
        return max;
    }

    private static double computeExpSum(double[] row, double max) {
        double sum = 0.0;
        for (double v : row) {
            sum += Math.exp(v - max);
        }
        return sum;
    }

    private static void applySoftmax(double[] input, double[] output, double max, double sum) {
        for (int j = 0; j < input.length; j++) {
            output[j] = Math.exp(input[j] - max) / sum;
        }
    }

    public static Tensor applySoftmaxBackward(Tensor dL_dSoftmax, Tensor softmaxOutput) {
        Tensor result = new Tensor(dL_dSoftmax.rows, dL_dSoftmax.cols);
        computeBackwardSoftmaxRows(dL_dSoftmax, softmaxOutput, result);
        return result;
    }

    private static void computeBackwardSoftmaxRows(Tensor grad, Tensor softmax, Tensor result) {
        for (int i = 0; i < grad.rows; i++) {
            result.data[i] = computeSoftmaxGradientRow(grad.data[i], softmax.data[i]);
        }
    }

    private static double[] computeSoftmaxGradientRow(double[] grad, double[] softmax) {
        int n = grad.length;
        double[] result = new double[n];
        for (int j = 0; j < n; j++) {
            result[j] = computeSoftmaxGradDot(grad, softmax, j);
        }
        return result;
    }

    private static double computeSoftmaxGradDot(double[] grad, double[] softmax, int j) {
        double sum = 0.0;
        for (int k = 0; k < grad.length; k++) {
            double jacobian = (j == k)
                    ? softmax[j] * (1 - softmax[j])
                    : -softmax[j] * softmax[k];
            sum += jacobian * grad[k];
        }
        return sum;
    }

    public Tensor add(Tensor other) {
        Tensor result = new Tensor(this.rows, this.cols);
        iterate((r, c) -> result.data[r][c] = this.data[r][c] + other.data[r][c]);
        return result;
    }

    public Tensor addBroadcastCols(Tensor colVector) {
        Tensor result = new Tensor(this.rows, this.cols);
        iterate((r, c) -> result.data[r][c] = this.data[r][c] + colVector.data[r][0]);
        return result;
    }

    public Tensor subtract(Tensor other) {
        Tensor result = new Tensor(this.rows, this.cols);
        iterate((r, c) -> result.data[r][c] = this.data[r][c] - other.data[r][c]);
        return result;
    }

    public double mseLoss(Tensor target) {
        double[] sum = new double[]{0.0};
        iterate((r, c) -> sum[0] += Math.pow(this.data[r][c] - target.data[r][c], 2));
        return sum[0] / (this.rows * this.cols);
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
        int[] index = new int[]{0};
        iterate((r, c) -> flat[index[0]++] = t.data[r][c], t);
        return flat;
    }

    public Tensor meanAlongRows() {
        Tensor sum = new Tensor(this.rows, 1);
        iterate((r, c) -> sum.data[r][0] += this.data[r][c]);
        return sum.divideScalar(this.cols);
    }

    public Tensor varianceAlongRows() {
        Tensor mean = this.meanAlongRows();
        Tensor result = new Tensor(this.rows, 1);

        iterate((r, c) -> {
            double diff = this.data[r][c] - mean.data[r][0];
            result.data[r][0] += diff * diff;
        });

        return result.divideScalar(cols);
    }

    public Tensor sumAlongRows() {
        Tensor result = new Tensor(this.rows, 1);
        iterate((r, c) -> result.data[r][0] += this.data[r][c]);
        return result;
    }

    public Tensor sumAlongCols() {
        Tensor result = new Tensor(1, this.cols);
        iterate((r, c) -> result.data[0][c] += this.data[r][c]);
        return result;
    }

    public Tensor divideBroadcastCols(Tensor colVector) {
        Tensor result = new Tensor(this.rows, this.cols);
        iterate((r, c) -> result.data[r][c] = this.data[r][c] / colVector.data[r][0]);
        return result;
    }

    public Tensor subtractBroadcastCols(Tensor colVector) {
        Tensor result = new Tensor(this.rows, this.cols);
        iterate((r, c) -> result.data[r][c] = this.data[r][c] - colVector.data[r][0]);
        return result;
    }

    public Tensor addBroadcastRows(Tensor rowVector) {
        Tensor result = new Tensor(this.rows, this.cols);
        iterate((r, c) -> result.data[r][c] = this.data[r][c] + rowVector.data[0][c]);
        return result;
    }

    public Tensor multiplyBroadcastRows(Tensor rowVector) {
        Tensor result = new Tensor(this.rows, this.cols);
        iterate((r, c) -> result.data[r][c] = this.data[r][c] * rowVector.data[0][c]);
        return result;
    }

    public Tensor sqrt() {
        Tensor result = new Tensor(this.rows, this.cols);
        iterate((r, c) -> result.data[r][c] = Math.sqrt(this.data[r][c]));
        return result;
    }

    public Tensor pow(double exponent) {
        Tensor result = new Tensor(this.rows, this.cols);
        iterate((r, c) -> result.data[r][c] = Math.pow(this.data[r][c], exponent));
        return result;
    }

    public static Tensor ones(int rows, int cols) {
        Tensor result = new Tensor(rows, cols);
        iterate((r, c) ->  result.data[r][c] = 1.0, result);
        return result;
    }

    public static Tensor zeros(int rows, int cols) {
        return new Tensor(rows, cols);
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
