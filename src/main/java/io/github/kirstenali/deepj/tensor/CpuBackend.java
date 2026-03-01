package io.github.kirstenali.deepj.tensor;

import java.util.Arrays;
import java.util.Random;

public final class CpuBackend implements TensorBackend {

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

    // Factories

    @Override
    public Tensor zeros(int rows, int cols) {
        return new Tensor(rows, cols);
    }

    @Override
    public Tensor ones(int rows, int cols) {
        Tensor result = new Tensor(rows, cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result.data[r][c] = 1.0;
        return result;
    }

    @Override
    public Tensor random(int rows, int cols, Random rand) {
        Tensor t = new Tensor(rows, cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                t.data[r][c] = rand.nextGaussian() * 0.1;
            }
        }
        return t;
    }

    @Override
    public Tensor causalMask(int size) {
        Tensor mask = new Tensor(size, size);
        for (int r = 0; r < size; r++) {
            for (int c = 0; c < size; c++) {
                mask.data[r][c] = (c > r) ? -1e9 : 0.0;
            }
        }
        return mask;
    }

    @Override
    public Tensor unflattenToTensor(double[] flat, int rows, int cols) {
        Tensor t = new Tensor(rows, cols);
        for (int i = 0; i < flat.length; i++) {
            t.data[i / cols][i % cols] = flat[i];
        }
        return t;
    }

    @Override
    public double[] flattenTensor(Tensor t) {
        double[] flat = new double[t.rows * t.cols];
        int idx = 0;
        for (int r = 0; r < t.rows; r++)
            for (int c = 0; c < t.cols; c++)
                flat[idx++] = t.data[r][c];
        return flat;
    }

    // Core ops

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        requireMatmulCompatible(a, b);
        Tensor result = new Tensor(a.rows, b.cols);

        for (int r = 0; r < a.rows; r++) {
            for (int c = 0; c < b.cols; c++) {
                double sum = 0.0;
                for (int k = 0; k < a.cols; k++) {
                    sum += a.data[r][k] * b.data[k][c];
                }
                result.data[r][c] = sum;
            }
        }
        return result;
    }

    @Override
    public Tensor add(Tensor a, Tensor b) {
        requireSameShape(a, b, "add");
        Tensor result = new Tensor(a.rows, a.cols);
        for (int r = 0; r < a.rows; r++)
            for (int c = 0; c < a.cols; c++)
                result.data[r][c] = a.data[r][c] + b.data[r][c];
        return result;
    }

    @Override
    public Tensor subtract(Tensor a, Tensor b) {
        requireSameShape(a, b, "subtract");
        Tensor result = new Tensor(a.rows, a.cols);
        for (int r = 0; r < a.rows; r++)
            for (int c = 0; c < a.cols; c++)
                result.data[r][c] = a.data[r][c] - b.data[r][c];
        return result;
    }

    @Override
    public Tensor multiply(Tensor a, Tensor b) {
        requireSameShape(a, b, "multiply");
        Tensor result = new Tensor(a.rows, a.cols);
        for (int r = 0; r < a.rows; r++)
            for (int c = 0; c < a.cols; c++)
                result.data[r][c] = a.data[r][c] * b.data[r][c];
        return result;
    }

    @Override
    public Tensor divide(Tensor a, Tensor b) {
        requireSameShape(a, b, "divide");
        Tensor result = new Tensor(a.rows, a.cols);
        for (int r = 0; r < a.rows; r++)
            for (int c = 0; c < a.cols; c++)
                result.data[r][c] = a.data[r][c] / b.data[r][c];
        return result;
    }

    // Broadcasts

    @Override
    public Tensor addRowVector(Tensor a, Tensor rowVector) {
        if (rowVector.rows != 1 || rowVector.cols != a.cols)
            throw new IllegalArgumentException("rowVector must be 1x" + a.cols);

        Tensor result = new Tensor(a.rows, a.cols);
        for (int r = 0; r < a.rows; r++) {
            for (int c = 0; c < a.cols; c++) {
                result.data[r][c] = a.data[r][c] + rowVector.data[0][c];
            }
        }
        return result;
    }

    @Override
    public Tensor addBroadcastRows(Tensor a, Tensor rowVector) {
        // same semantics as addRowVector
        return addRowVector(a, rowVector);
    }

    @Override
    public Tensor multiplyBroadcastRows(Tensor a, Tensor rowVector) {
        if (rowVector.rows != 1 || rowVector.cols != a.cols)
            throw new IllegalArgumentException("rowVector must be 1x" + a.cols);

        Tensor result = new Tensor(a.rows, a.cols);
        for (int r = 0; r < a.rows; r++)
            for (int c = 0; c < a.cols; c++)
                result.data[r][c] = a.data[r][c] * rowVector.data[0][c];
        return result;
    }

    @Override
    public Tensor addBroadcastCols(Tensor a, Tensor colVector) {
        if (colVector.cols != 1 || colVector.rows != a.rows)
            throw new IllegalArgumentException("colVector must be " + a.rows + "x1");

        Tensor result = new Tensor(a.rows, a.cols);
        for (int r = 0; r < a.rows; r++)
            for (int c = 0; c < a.cols; c++)
                result.data[r][c] = a.data[r][c] + colVector.data[r][0];
        return result;
    }

    @Override
    public Tensor subtractBroadcastCols(Tensor a, Tensor colVector) {
        if (colVector.cols != 1 || colVector.rows != a.rows)
            throw new IllegalArgumentException("colVector must be " + a.rows + "x1");

        Tensor result = new Tensor(a.rows, a.cols);
        for (int r = 0; r < a.rows; r++)
            for (int c = 0; c < a.cols; c++)
                result.data[r][c] = a.data[r][c] - colVector.data[r][0];
        return result;
    }

    @Override
    public Tensor multiplyBroadcastCols(Tensor a, Tensor colVector) {
        if (colVector.cols != 1 || colVector.rows != a.rows)
            throw new IllegalArgumentException("colVector must be " + a.rows + "x1");

        Tensor result = new Tensor(a.rows, a.cols);
        for (int r = 0; r < a.rows; r++)
            for (int c = 0; c < a.cols; c++)
                result.data[r][c] = a.data[r][c] * colVector.data[r][0];
        return result;
    }

    @Override
    public Tensor divideBroadcastCols(Tensor a, Tensor colVector) {
        if (colVector.cols != 1 || colVector.rows != a.rows)
            throw new IllegalArgumentException("colVector must be " + a.rows + "x1");

        Tensor result = new Tensor(a.rows, a.cols);
        for (int r = 0; r < a.rows; r++)
            for (int c = 0; c < a.cols; c++)
                result.data[r][c] = a.data[r][c] / colVector.data[r][0];
        return result;
    }

    // Reductions / stats

    @Override
    public Tensor sumRows(Tensor a) {
        // sums across rows -> 1 x cols
        Tensor result = new Tensor(1, a.cols);
        for (int r = 0; r < a.rows; r++)
            for (int c = 0; c < a.cols; c++)
                result.data[0][c] += a.data[r][c];
        return result;
    }

    @Override
    public Tensor sumAlongRows(Tensor a) {
        // sums across cols -> rows x 1
        Tensor result = new Tensor(a.rows, 1);
        for (int r = 0; r < a.rows; r++)
            for (int c = 0; c < a.cols; c++)
                result.data[r][0] += a.data[r][c];
        return result;
    }

    @Override
    public Tensor sumAlongCols(Tensor a) {
        // sums across rows -> 1 x cols
        return sumRows(a);
    }

    @Override
    public Tensor meanAlongRows(Tensor a) {
        Tensor sum = new Tensor(a.rows, 1);
        for (int r = 0; r < a.rows; r++)
            for (int c = 0; c < a.cols; c++)
                sum.data[r][0] += a.data[r][c];
        return divideScalar(sum, a.cols);
    }

    @Override
    public Tensor varianceAlongRows(Tensor a) {
        Tensor mean = meanAlongRows(a);
        Tensor result = new Tensor(a.rows, 1);

        for (int r = 0; r < a.rows; r++) {
            for (int c = 0; c < a.cols; c++) {
                double diff = a.data[r][c] - mean.data[r][0];
                result.data[r][0] += diff * diff;
            }
        }
        return divideScalar(result, a.cols);
    }

    // Unary ops

    @Override
    public Tensor transpose(Tensor a) {
        Tensor result = new Tensor(a.cols, a.rows);
        for (int r = 0; r < a.rows; r++)
            for (int c = 0; c < a.cols; c++)
                result.data[c][r] = a.data[r][c];
        return result;
    }

    @Override
    public Tensor clamp(Tensor a, double min, double max) {
        Tensor result = new Tensor(a.rows, a.cols);
        for (int r = 0; r < a.rows; r++) {
            for (int c = 0; c < a.cols; c++) {
                double v = a.data[r][c];
                if (v < min) v = min;
                if (v > max) v = max;
                result.data[r][c] = v;
            }
        }
        return result;
    }

    @Override
    public Tensor sqrt(Tensor a) {
        Tensor result = new Tensor(a.rows, a.cols);
        for (int r = 0; r < a.rows; r++)
            for (int c = 0; c < a.cols; c++)
                result.data[r][c] = Math.sqrt(a.data[r][c]);
        return result;
    }

    @Override
    public Tensor pow(Tensor a, double exponent) {
        Tensor result = new Tensor(a.rows, a.cols);
        for (int r = 0; r < a.rows; r++)
            for (int c = 0; c < a.cols; c++)
                result.data[r][c] = Math.pow(a.data[r][c], exponent);
        return result;
    }

    // Scalar ops

    @Override
    public Tensor multiplyScalar(Tensor a, double scalar) {
        Tensor result = new Tensor(a.rows, a.cols);
        for (int r = 0; r < a.rows; r++)
            for (int c = 0; c < a.cols; c++)
                result.data[r][c] = a.data[r][c] * scalar;
        return result;
    }

    @Override
    public Tensor addScalar(Tensor a, double scalar) {
        Tensor result = new Tensor(a.rows, a.cols);
        for (int r = 0; r < a.rows; r++)
            for (int c = 0; c < a.cols; c++)
                result.data[r][c] = a.data[r][c] + scalar;
        return result;
    }

    @Override
    public Tensor divideScalar(Tensor a, double scalar) {
        Tensor result = new Tensor(a.rows, a.cols);
        for (int r = 0; r < a.rows; r++)
            for (int c = 0; c < a.cols; c++)
                result.data[r][c] = a.data[r][c] / scalar;
        return result;
    }

    // Loss / sums

    @Override
    public double sum(Tensor a) {
        double s = 0.0;
        for (int r = 0; r < a.rows; r++)
            for (int c = 0; c < a.cols; c++)
                s += a.data[r][c];
        return s;
    }

    @Override
    public double mseLoss(Tensor prediction, Tensor target) {
        requireSameShape(prediction, target, "mseLoss");
        double s = 0.0;
        for (int r = 0; r < prediction.rows; r++) {
            for (int c = 0; c < prediction.cols; c++) {
                double d = prediction.data[r][c] - target.data[r][c];
                s += d * d;
            }
        }
        return s / (prediction.rows * prediction.cols);
    }

    // Softmax

    @Override
    public Tensor softmaxRows(Tensor logits) {
        Tensor result = new Tensor(logits.rows, logits.cols);
        for (int i = 0; i < logits.rows; i++) {
            computeSoftmaxRow(logits.data[i], result.data[i]);
        }
        return result;
    }

    private static void computeSoftmaxRow(double[] inputRow, double[] outputRow) {
        double max = findMax(inputRow);
        double sum = computeExpSum(inputRow, max);
        for (int j = 0; j < inputRow.length; j++) {
            outputRow[j] = Math.exp(inputRow[j] - max) / sum;
        }
    }

    private static double findMax(double[] row) {
        return Arrays.stream(row).max().orElse(Double.NEGATIVE_INFINITY);
    }

    private static double computeExpSum(double[] row, double max) {
        return Arrays.stream(row)
                .map(v -> Math.exp(v - max))
                .sum();
    }

    @Override
    public Tensor softmaxBackward(Tensor upstreamGrad, Tensor softmaxOutput) {
        requireSameShape(upstreamGrad, softmaxOutput, "softmaxBackward");

        Tensor gradWrtLogits = new Tensor(upstreamGrad.rows, upstreamGrad.cols);

        for (int row = 0; row < upstreamGrad.rows; row++) {
            double[] gradRow = upstreamGrad.data[row];
            double[] probRow = softmaxOutput.data[row];
            double[] outRow  = gradWrtLogits.data[row];

            double weightedSum = dotProduct(gradRow, probRow);

            for (int j = 0; j < gradRow.length; j++) {
                outRow[j] = probRow[j] * (gradRow[j] - weightedSum);
            }
        }
        return gradWrtLogits;
    }

    private static double dotProduct(double[] a, double[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }

    // Debug

    @Override
    public void print(Tensor t, String label) {
        System.out.println(label);
        for (double[] row : t.data) {
            for (double val : row) {
                System.out.printf("%.4f ", val);
            }
            System.out.println();
        }
    }
}