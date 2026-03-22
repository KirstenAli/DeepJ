package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;

import java.util.Random;

public class Tensor {
    public final double[][] data;
    public final int rows, cols;

    private static volatile TensorBackend BACKEND = new CpuBackend(); // default

    public static void setBackend(TensorBackend backend) {
        if (backend == null) throw new IllegalArgumentException("backend cannot be null");
        BACKEND = backend;
    }

    public static TensorBackend backend() {
        return BACKEND;
    }

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

    // ---- instance ops delegate ----
    public Tensor matmul(Tensor other) { return backend().matmul(this, other); }
    public Tensor add(Tensor other) { return backend().add(this, other); }
    public Tensor subtract(Tensor other) { return backend().subtract(this, other); }
    public Tensor multiply(Tensor other) { return backend().multiply(this, other); }
    public Tensor divide(Tensor other) { return backend().divide(this, other); }

    public Tensor addRowVector(Tensor rowVector) { return backend().addRowVector(this, rowVector); }
    public Tensor sumRows() { return backend().sumRows(this); }

    public Tensor clamp(double min, double max) { return backend().clamp(this, min, max); }
    public Tensor transpose() { return backend().transpose(this); }

    public Tensor multiplyScalar(double s) { return backend().multiplyScalar(this, s); }
    public Tensor addScalar(double s) { return backend().addScalar(this, s); }
    public Tensor divideScalar(double s) { return backend().divideScalar(this, s); }

    public Tensor meanAlongRows() { return backend().meanAlongRows(this); }
    public Tensor varianceAlongRows() { return backend().varianceAlongRows(this); }

    public Tensor sumAlongRows() { return backend().sumAlongRows(this); }
    public Tensor sumAlongCols() { return backend().sumAlongCols(this); }

    public Tensor addBroadcastCols(Tensor colVector) { return backend().addBroadcastCols(this, colVector); }
    public Tensor divideBroadcastCols(Tensor colVector) { return backend().divideBroadcastCols(this, colVector); }
    public Tensor subtractBroadcastCols(Tensor colVector) { return backend().subtractBroadcastCols(this, colVector); }
    public Tensor multiplyBroadcastCols(Tensor colVector) { return backend().multiplyBroadcastCols(this, colVector); }

    public Tensor addBroadcastRows(Tensor rowVector) { return backend().addBroadcastRows(this, rowVector); }
    public Tensor multiplyBroadcastRows(Tensor rowVector) { return backend().multiplyBroadcastRows(this, rowVector); }

    public Tensor sqrt() { return backend().sqrt(this); }
    public Tensor pow(double exponent) { return backend().pow(this, exponent); }

    public double sum() { return backend().sum(this); }
    public double sumAbs() {return backend().sumAbs(this); }

    public void print(String label) { backend().print(this, label); }

    // ---- static ops delegate ----
    public static Tensor zeros(int rows, int cols) { return backend().zeros(rows, cols); }
    public static Tensor ones(int rows, int cols) { return backend().ones(rows, cols); }
    public static Tensor random(int rows, int cols, Random rand) { return backend().random(rows, cols, rand); }
    public static Tensor causalMask(int size) { return backend().causalMask(size); }

    public static Tensor unflattenToTensor(double[] flat, int rows, int cols) {return backend().unflattenToTensor(flat, rows, cols);}
    public static double[] flattenTensor(Tensor t) { return backend().flattenTensor(t); }

    public static void requireSameShape(Tensor a, Tensor b, String op) {
        if (a.rows != b.rows || a.cols != b.cols) {
            throw new IllegalArgumentException(
                    "Shape mismatch for " + op + ": " + a.rows + "x" + a.cols +
                            " vs " + b.rows + "x" + b.cols);
        }
    }
}