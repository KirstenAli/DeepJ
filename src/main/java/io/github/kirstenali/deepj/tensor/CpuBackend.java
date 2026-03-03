package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;

import java.util.Random;

import static io.github.kirstenali.deepj.tensor.Tensor.requireSameShape;

public final class CpuBackend implements TensorBackend {

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
        DeepJExecutor.range(0, rows).parallel().forEach(r -> {
            double[] rr = result.data[r];
            for (int c = 0; c < cols; c++) {
                rr[c] = 1.0;
            }
        });
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
        DeepJExecutor.range(0, size).parallel().forEach(r -> {
            double[] mr = mask.data[r];
            for (int c = 0; c < size; c++) {
                mr[c] = (c > r) ? -1e9 : 0.0;
            }
        });
        return mask;
    }

    @Override
    public Tensor unflattenToTensor(double[] flat, int rows, int cols) {
        Tensor t = new Tensor(rows, cols);
        DeepJExecutor.range(0, flat.length).parallel().forEach(i -> {
            t.data[i / cols][i % cols] = flat[i];
        });
        return t;
    }

    @Override
    public double[] flattenTensor(Tensor t) {
        double[] flat = new double[t.rows * t.cols];
        DeepJExecutor.range(0, t.rows).parallel().forEach(r -> {
            int base = r * t.cols;
            double[] tr = t.data[r];
            for (int c = 0; c < t.cols; c++) {
                flat[base + c] = tr[c];
            }
        });
        return flat;
    }

    // Core ops

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        requireMatmulCompatible(a, b);
        Tensor result = new Tensor(a.rows, b.cols);

        DeepJExecutor.range(0, a.rows).parallel().forEach(r -> {
            double[] aRow = a.data[r];
            double[] outRow = result.data[r];
            for (int c = 0; c < b.cols; c++) {
                double sum = 0.0;
                for (int k = 0; k < a.cols; k++) {
                    sum += aRow[k] * b.data[k][c];
                }
                outRow[c] = sum;
            }
        });

        return result;
    }

    @Override
    public Tensor add(Tensor a, Tensor b) {
        requireSameShape(a, b, "add");
        Tensor result = new Tensor(a.rows, a.cols);

        DeepJExecutor.range(0, a.rows).parallel().forEach(r -> {
            double[] ar = a.data[r], br = b.data[r], rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                rr[c] = ar[c] + br[c];
            }
        });

        return result;
    }

    @Override
    public Tensor subtract(Tensor a, Tensor b) {
        requireSameShape(a, b, "subtract");
        Tensor result = new Tensor(a.rows, a.cols);

        DeepJExecutor.range(0, a.rows).parallel().forEach(r -> {
            double[] ar = a.data[r], br = b.data[r], rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                rr[c] = ar[c] - br[c];
            }
        });

        return result;
    }

    @Override
    public Tensor multiply(Tensor a, Tensor b) {
        requireSameShape(a, b, "multiply");
        Tensor result = new Tensor(a.rows, a.cols);

        DeepJExecutor.range(0, a.rows).parallel().forEach(r -> {
            double[] ar = a.data[r], br = b.data[r], rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                rr[c] = ar[c] * br[c];
            }
        });

        return result;
    }

    @Override
    public Tensor divide(Tensor a, Tensor b) {
        requireSameShape(a, b, "divide");
        Tensor result = new Tensor(a.rows, a.cols);

        DeepJExecutor.range(0, a.rows).parallel().forEach(r -> {
            double[] ar = a.data[r], br = b.data[r], rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                rr[c] = ar[c] / br[c];
            }
        });

        return result;
    }

    // Broadcasts

    @Override
    public Tensor addRowVector(Tensor a, Tensor rowVector) {
        if (rowVector.rows != 1 || rowVector.cols != a.cols)
            throw new IllegalArgumentException("rowVector must be 1x" + a.cols);

        Tensor result = new Tensor(a.rows, a.cols);
        double[] rv = rowVector.data[0];

        DeepJExecutor.range(0, a.rows).parallel().forEach(r -> {
            double[] ar = a.data[r];
            double[] rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                rr[c] = ar[c] + rv[c];
            }
        });

        return result;
    }

    @Override
    public Tensor addBroadcastRows(Tensor a, Tensor rowVector) {
        return addRowVector(a, rowVector);
    }

    @Override
    public Tensor multiplyBroadcastRows(Tensor a, Tensor rowVector) {
        if (rowVector.rows != 1 || rowVector.cols != a.cols)
            throw new IllegalArgumentException("rowVector must be 1x" + a.cols);

        Tensor result = new Tensor(a.rows, a.cols);
        double[] rv = rowVector.data[0];

        DeepJExecutor.range(0, a.rows).parallel().forEach(r -> {
            double[] ar = a.data[r];
            double[] rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                rr[c] = ar[c] * rv[c];
            }
        });

        return result;
    }

    @Override
    public Tensor addBroadcastCols(Tensor a, Tensor colVector) {
        if (colVector.cols != 1 || colVector.rows != a.rows)
            throw new IllegalArgumentException("colVector must be " + a.rows + "x1");

        Tensor result = new Tensor(a.rows, a.cols);

        DeepJExecutor.range(0, a.rows).parallel().forEach(r -> {
            double v = colVector.data[r][0];
            double[] ar = a.data[r];
            double[] rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                rr[c] = ar[c] + v;
            }
        });

        return result;
    }

    @Override
    public Tensor subtractBroadcastCols(Tensor a, Tensor colVector) {
        if (colVector.cols != 1 || colVector.rows != a.rows)
            throw new IllegalArgumentException("colVector must be " + a.rows + "x1");

        Tensor result = new Tensor(a.rows, a.cols);

        DeepJExecutor.range(0, a.rows).parallel().forEach(r -> {
            double v = colVector.data[r][0];
            double[] ar = a.data[r];
            double[] rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                rr[c] = ar[c] - v;
            }
        });

        return result;
    }

    @Override
    public Tensor multiplyBroadcastCols(Tensor a, Tensor colVector) {
        if (colVector.cols != 1 || colVector.rows != a.rows)
            throw new IllegalArgumentException("colVector must be " + a.rows + "x1");

        Tensor result = new Tensor(a.rows, a.cols);

        DeepJExecutor.range(0, a.rows).parallel().forEach(r -> {
            double v = colVector.data[r][0];
            double[] ar = a.data[r];
            double[] rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                rr[c] = ar[c] * v;
            }
        });

        return result;
    }

    @Override
    public Tensor divideBroadcastCols(Tensor a, Tensor colVector) {
        if (colVector.cols != 1 || colVector.rows != a.rows)
            throw new IllegalArgumentException("colVector must be " + a.rows + "x1");

        Tensor result = new Tensor(a.rows, a.cols);

        DeepJExecutor.range(0, a.rows).parallel().forEach(r -> {
            double v = colVector.data[r][0];
            double[] ar = a.data[r];
            double[] rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                rr[c] = ar[c] / v;
            }
        });

        return result;
    }

    // Reductions / stats

    @Override
    public Tensor sumRows(Tensor a) {
        Tensor result = new Tensor(1, a.cols);
        for (int r = 0; r < a.rows; r++)
            for (int c = 0; c < a.cols; c++)
                result.data[0][c] += a.data[r][c];
        return result;
    }

    @Override
    public Tensor sumAlongRows(Tensor a) {
        Tensor result = new Tensor(a.rows, 1);

        DeepJExecutor.range(0, a.rows).parallel().forEach(r -> {
            double sum = 0.0;
            double[] ar = a.data[r];
            for (int c = 0; c < a.cols; c++) {
                sum += ar[c];
            }
            result.data[r][0] = sum;
        });

        return result;
    }

    @Override
    public Tensor sumAlongCols(Tensor a) {
        return sumRows(a);
    }

    @Override
    public Tensor meanAlongRows(Tensor a) {
        Tensor sum = new Tensor(a.rows, 1);

        DeepJExecutor.range(0, a.rows).parallel().forEach(r -> {
            double s = 0.0;
            double[] ar = a.data[r];
            for (int c = 0; c < a.cols; c++) {
                s += ar[c];
            }
            sum.data[r][0] = s;
        });

        return divideScalar(sum, a.cols);
    }

    @Override
    public Tensor varianceAlongRows(Tensor a) {
        Tensor mean = meanAlongRows(a);
        Tensor result = new Tensor(a.rows, 1);

        DeepJExecutor.range(0, a.rows).parallel().forEach(r -> {
            double m = mean.data[r][0];
            double acc = 0.0;
            double[] ar = a.data[r];
            for (int c = 0; c < a.cols; c++) {
                double diff = ar[c] - m;
                acc += diff * diff;
            }
            result.data[r][0] = acc;
        });

        return divideScalar(result, a.cols);
    }

    // Unary ops

    @Override
    public Tensor transpose(Tensor a) {
        Tensor result = new Tensor(a.cols, a.rows);

        DeepJExecutor.range(0, a.rows).parallel().forEach(r -> {
            double[] ar = a.data[r];
            for (int c = 0; c < a.cols; c++) {
                result.data[c][r] = ar[c];
            }
        });

        return result;
    }

    @Override
    public Tensor clamp(Tensor a, double min, double max) {
        Tensor result = new Tensor(a.rows, a.cols);

        DeepJExecutor.range(0, a.rows).parallel().forEach(r -> {
            double[] ar = a.data[r];
            double[] rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                double v = ar[c];
                if (v < min) v = min;
                if (v > max) v = max;
                rr[c] = v;
            }
        });

        return result;
    }

    @Override
    public Tensor sqrt(Tensor a) {
        Tensor result = new Tensor(a.rows, a.cols);

        DeepJExecutor.range(0, a.rows).parallel().forEach(r -> {
            double[] ar = a.data[r];
            double[] rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                rr[c] = Math.sqrt(ar[c]);
            }
        });

        return result;
    }

    @Override
    public Tensor pow(Tensor a, double exponent) {
        Tensor result = new Tensor(a.rows, a.cols);

        DeepJExecutor.range(0, a.rows).parallel().forEach(r -> {
            double[] ar = a.data[r];
            double[] rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                rr[c] = Math.pow(ar[c], exponent);
            }
        });

        return result;
    }

    // Scalar ops

    @Override
    public Tensor multiplyScalar(Tensor a, double scalar) {
        Tensor result = new Tensor(a.rows, a.cols);

        DeepJExecutor.range(0, a.rows).parallel().forEach(r -> {
            double[] ar = a.data[r];
            double[] rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                rr[c] = ar[c] * scalar;
            }
        });

        return result;
    }

    @Override
    public Tensor addScalar(Tensor a, double scalar) {
        Tensor result = new Tensor(a.rows, a.cols);

        DeepJExecutor.range(0, a.rows).parallel().forEach(r -> {
            double[] ar = a.data[r];
            double[] rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                rr[c] = ar[c] + scalar;
            }
        });

        return result;
    }

    @Override
    public Tensor divideScalar(Tensor a, double scalar) {
        Tensor result = new Tensor(a.rows, a.cols);

        DeepJExecutor.range(0, a.rows).parallel().forEach(r -> {
            double[] ar = a.data[r];
            double[] rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                rr[c] = ar[c] / scalar;
            }
        });

        return result;
    }

    // Loss

    @Override
    public double sum(Tensor a) {
        double s = 0.0;
        for (int r = 0; r < a.rows; r++)
            for (int c = 0; c < a.cols; c++)
                s += a.data[r][c];
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