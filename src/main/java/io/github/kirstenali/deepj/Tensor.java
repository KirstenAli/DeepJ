package io.github.kirstenali.deepj;

import java.util.Arrays;
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
        for (int i = 0; i < rows; i++) {
            if (data[i].length != cols)
                throw new IllegalArgumentException("All rows must have the same length (expected " + cols + ")");
            System.arraycopy(data[i], 0, this.data[i], 0, cols);
        }
    }

    private static void requireSameShape(Tensor a, Tensor b, String op) {
        if (a.rows != b.rows || a.cols != b.cols) {
            throw new IllegalArgumentException(
                    "Shape mismatch for " + op + ": " + a.rows + "x" + a.cols +
                            " vs " + b.rows + "x" + b.cols);
        }
    }

    private static void requireMatmulCompatible(Tensor a, Tensor b) {
        if (a.cols != b.rows) {
            throw new IllegalArgumentException(
                    "Shape mismatch for matmul: " + a.rows + "x" + a.cols +
                            " cannot be multiplied by " + b.rows + "x" + b.cols);
        }
    }

    public static Tensor random(int rows, int cols, Random rand) {
        Tensor t = new Tensor(rows, cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                t.data[r][c] = rand.nextGaussian() * 0.1;
            }
        }
        return t;
    }

    public Tensor matmul(Tensor other) {
        requireMatmulCompatible(this, other);
        Tensor result = new Tensor(this.rows, other.cols);

        for (int r = 0; r < this.rows; r++) {
            for (int c = 0; c < other.cols; c++) {
                double sum = 0.0;
                for (int k = 0; k < this.cols; k++) {
                    sum += this.data[r][k] * other.data[k][c];
                }
                result.data[r][c] = sum;
            }
        }

        return result;
    }


    public Tensor addRowVector(Tensor rowVector) {
        if (rowVector.rows != 1 || rowVector.cols != this.cols)
            throw new IllegalArgumentException("rowVector must be 1x" + this.cols);
        Tensor result = new Tensor(this.rows, this.cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result.data[r][c] = this.data[r][c] + rowVector.data[0][c];
            }
        }
        return result;
    }

    public Tensor sumRows() {
        Tensor result = new Tensor(1, this.cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result.data[0][c] += this.data[r][c];
            }
        }
        return result;
    }

    public Tensor divide(Tensor other) {
        requireSameShape(this, other, "divide");
        Tensor result = new Tensor(rows, cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result.data[r][c] = this.data[r][c] / other.data[r][c];
            }
        }
        return result;
    }

    public Tensor clamp(double min, double max) {
        Tensor result = new Tensor(rows, cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double v = this.data[r][c];
                if (v < min) v = min;
                if (v > max) v = max;
                result.data[r][c] = v;
            }
        }
        return result;
    }


    public Tensor multiply(Tensor other) {
        requireSameShape(this, other, "multiply");
        Tensor result = new Tensor(this.rows, this.cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result.data[r][c] = this.data[r][c] * other.data[r][c];
            }
        }
        return result;
    }

    public Tensor multiplyBroadcastCols(Tensor colVector) {
        if (colVector.cols != 1 || colVector.rows != this.rows)
            throw new IllegalArgumentException("colVector must be " + this.rows + "x1");
        Tensor result = new Tensor(this.rows, this.cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result.data[r][c] = this.data[r][c] * colVector.data[r][0];
            }
        }
        return result;
    }

    public double sum() {
        double sum = 0.0;
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                sum += data[r][c];
        return sum;
    }

    public Tensor transpose() {
        Tensor result = new Tensor(this.cols, this.rows);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result.data[c][r] = this.data[r][c];
        return result;
    }

    public Tensor multiplyScalar(double scalar) {
        Tensor result = new Tensor(this.rows, this.cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result.data[r][c] = this.data[r][c] * scalar;
        return result;
    }

    public Tensor addScalar(double scalar) {
        Tensor result = new Tensor(this.rows, this.cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result.data[r][c] = this.data[r][c] + scalar;
        return result;
    }

    public Tensor divideScalar(double scalar) {
        Tensor result = new Tensor(this.rows, this.cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result.data[r][c] = this.data[r][c] / scalar;
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
        softmax(inputRow, outputRow, max, sum);
    }

    private static double findMax(double[] row) {
        return Arrays.stream(row).max().orElse(Double.NEGATIVE_INFINITY);
    }

    private static double computeExpSum(double[] row, double max) {
        return Arrays.stream(row)
                .map(v -> Math.exp(v - max))
                .sum();
    }

    private static void softmax(double[] input, double[] output, double max, double sum) {
        for (int j = 0; j < input.length; j++) {
            output[j] = Math.exp(input[j] - max) / sum;
        }
    }

    public static Tensor softmaxBackward(Tensor upstreamGrad,
                                         Tensor softmaxOutput) {

        Tensor gradWrtLogits = new Tensor(upstreamGrad.rows, upstreamGrad.cols);

        for (int row = 0; row < upstreamGrad.rows; row++) {
            double[] gradRow   = upstreamGrad.data[row];
            double[] probRow   = softmaxOutput.data[row];
            double[] outRow    = gradWrtLogits.data[row];

            double weightedSum = dotProduct(gradRow, probRow);

            fillLogitGradRow(gradRow, probRow, outRow, weightedSum);
        }
        return gradWrtLogits;
    }

    private static double dotProduct(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }

    private static void fillLogitGradRow(double[] gradRow,
                                         double[] probRow,
                                         double[] outRow,
                                         double weightedSum) {
        for (int j = 0; j < gradRow.length; j++) {
            outRow[j] = probRow[j] * (gradRow[j] - weightedSum);
        }
    }

    public Tensor add(Tensor other) {
        requireSameShape(this, other, "add");
        Tensor result = new Tensor(this.rows, this.cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result.data[r][c] = this.data[r][c] + other.data[r][c];
        return result;
    }

    public Tensor addBroadcastCols(Tensor colVector) {
        if (colVector.cols != 1 || colVector.rows != this.rows)
            throw new IllegalArgumentException("colVector must be " + this.rows + "x1");
        Tensor result = new Tensor(this.rows, this.cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result.data[r][c] = this.data[r][c] + colVector.data[r][0];
        return result;
    }

    public Tensor subtract(Tensor other) {
        requireSameShape(this, other, "subtract");
        Tensor result = new Tensor(this.rows, this.cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result.data[r][c] = this.data[r][c] - other.data[r][c];
        return result;
    }

    public double mseLoss(Tensor target) {
        requireSameShape(this, target, "mseLoss");
        double sum = 0.0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double d = this.data[r][c] - target.data[r][c];
                sum += d * d;
            }
        }
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
        int idx = 0;
        for (int r = 0; r < t.rows; r++)
            for (int c = 0; c < t.cols; c++)
                flat[idx++] = t.data[r][c];
        return flat;
    }

    public Tensor meanAlongRows() {
        Tensor sum = new Tensor(this.rows, 1);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                sum.data[r][0] += this.data[r][c];
        return sum.divideScalar(this.cols);
    }

    public Tensor varianceAlongRows() {
        Tensor mean = this.meanAlongRows();
        Tensor result = new Tensor(this.rows, 1);

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double diff = this.data[r][c] - mean.data[r][0];
                result.data[r][0] += diff * diff;
            }
        }

        return result.divideScalar(cols);
    }

    public Tensor sumAlongRows() {
        Tensor result = new Tensor(this.rows, 1);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result.data[r][0] += this.data[r][c];
        return result;
    }

    public Tensor sumAlongCols() {
        Tensor result = new Tensor(1, this.cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result.data[0][c] += this.data[r][c];
        return result;
    }

    public Tensor divideBroadcastCols(Tensor colVector) {
        if (colVector.cols != 1 || colVector.rows != this.rows)
            throw new IllegalArgumentException("colVector must be " + this.rows + "x1");
        Tensor result = new Tensor(this.rows, this.cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result.data[r][c] = this.data[r][c] / colVector.data[r][0];
        return result;
    }

    public Tensor subtractBroadcastCols(Tensor colVector) {
        if (colVector.cols != 1 || colVector.rows != this.rows)
            throw new IllegalArgumentException("colVector must be " + this.rows + "x1");
        Tensor result = new Tensor(this.rows, this.cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result.data[r][c] = this.data[r][c] - colVector.data[r][0];
        return result;
    }

    public Tensor addBroadcastRows(Tensor rowVector) {
        if (rowVector.rows != 1 || rowVector.cols != this.cols)
            throw new IllegalArgumentException("rowVector must be 1x" + this.cols);
        Tensor result = new Tensor(this.rows, this.cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result.data[r][c] = this.data[r][c] + rowVector.data[0][c];
        return result;
    }

    public Tensor multiplyBroadcastRows(Tensor rowVector) {
        if (rowVector.rows != 1 || rowVector.cols != this.cols)
            throw new IllegalArgumentException("rowVector must be 1x" + this.cols);
        Tensor result = new Tensor(this.rows, this.cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result.data[r][c] = this.data[r][c] * rowVector.data[0][c];
        return result;
    }

    public Tensor sqrt() {
        Tensor result = new Tensor(this.rows, this.cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result.data[r][c] = Math.sqrt(this.data[r][c]);
        return result;
    }

    public Tensor pow(double exponent) {
        Tensor result = new Tensor(this.rows, this.cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result.data[r][c] = Math.pow(this.data[r][c], exponent);
        return result;
    }

    public static Tensor ones(int rows, int cols) {
        Tensor result = new Tensor(rows, cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result.data[r][c] = 1.0;
        return result;
    }

    public static Tensor zeros(int rows, int cols) {
        return new Tensor(rows, cols);
    }

    public static Tensor causalMask(int size) {
        Tensor mask = new Tensor(size, size);
        for (int r = 0; r < size; r++) {
            for (int c = 0; c < size; c++) {
                mask.data[r][c] = (c > r) ? -1e9 : 0.0;
            }
        }
        return mask;
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
