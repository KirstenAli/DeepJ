package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;

import java.util.Random;

public final class MetalBackend implements TensorBackend {

    private final CpuBackend cpuFallback = new CpuBackend();

    private static final int MATMUL_GPU_THRESHOLD = 128 * 128 * 64;

    private static void requireMatmulCompatible(Tensor a, Tensor b) {
        if (a.cols != b.rows) {
            throw new IllegalArgumentException(
                    "Shape mismatch for matmul: " + a.rows + "x" + a.cols +
                            " cannot be multiplied by " + b.rows + "x" + b.cols);
        }
    }

    private static boolean useGpuMatmul(Tensor a, Tensor b) {
        long work = (long) a.rows * a.cols * b.cols;
        return MetalNative.AVAILABLE && work >= MATMUL_GPU_THRESHOLD;
    }

    @Override
    public Tensor zeros(int rows, int cols) {
        return cpuFallback.zeros(rows, cols);
    }

    @Override
    public Tensor ones(int rows, int cols) {
        return cpuFallback.ones(rows, cols);
    }

    @Override
    public Tensor random(int rows, int cols, Random rand) {
        return cpuFallback.random(rows, cols, rand);
    }

    @Override
    public Tensor causalMask(int size) {
        return cpuFallback.causalMask(size);
    }

    @Override
    public Tensor unflattenToTensor(double[] flat, int rows, int cols) {
        return cpuFallback.unflattenToTensor(flat, rows, cols);
    }

    @Override
    public double[] flattenTensor(Tensor t) {
        return cpuFallback.flattenTensor(t);
    }

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        requireMatmulCompatible(a, b);
        if (!useGpuMatmul(a, b)) return cpuFallback.matmul(a, b);

        float[] pa = TensorAdapters.packF32(a);
        float[] pb = TensorAdapters.packF32(b);
        float[] po = new float[a.rows * b.cols];
        MetalNative.matmulF32(pa, pb, po, a.rows, b.cols, a.cols);
        return TensorAdapters.unpackF32(po, a.rows, b.cols);
    }

    @Override
    public Tensor add(Tensor a, Tensor b) {
        return cpuFallback.add(a, b);
    }

    @Override
    public Tensor subtract(Tensor a, Tensor b) {
        return cpuFallback.subtract(a, b);
    }

    @Override
    public Tensor multiply(Tensor a, Tensor b) {
        return cpuFallback.multiply(a, b);
    }

    @Override
    public Tensor divide(Tensor a, Tensor b) {
        return cpuFallback.divide(a, b);
    }

    @Override
    public Tensor addRowVector(Tensor a, Tensor rowVector) {
        return cpuFallback.addRowVector(a, rowVector);
    }

    @Override
    public Tensor addBroadcastRows(Tensor a, Tensor rowVector) {
        return cpuFallback.addBroadcastRows(a, rowVector);
    }

    @Override
    public Tensor multiplyBroadcastRows(Tensor a, Tensor rowVector) {
        return cpuFallback.multiplyBroadcastRows(a, rowVector);
    }

    @Override
    public Tensor addBroadcastCols(Tensor a, Tensor colVector) {
        return cpuFallback.addBroadcastCols(a, colVector);
    }

    @Override
    public Tensor subtractBroadcastCols(Tensor a, Tensor colVector) {
        return cpuFallback.subtractBroadcastCols(a, colVector);
    }

    @Override
    public Tensor multiplyBroadcastCols(Tensor a, Tensor colVector) {
        return cpuFallback.multiplyBroadcastCols(a, colVector);
    }

    @Override
    public Tensor divideBroadcastCols(Tensor a, Tensor colVector) {
        return cpuFallback.divideBroadcastCols(a, colVector);
    }

    @Override
    public Tensor sumRows(Tensor a) {
        return cpuFallback.sumRows(a);
    }

    @Override
    public Tensor sumAlongRows(Tensor a) {
        return cpuFallback.sumAlongRows(a);
    }

    @Override
    public Tensor sumAlongCols(Tensor a) {
        return cpuFallback.sumAlongCols(a);
    }

    @Override
    public Tensor meanAlongRows(Tensor a) {
        return cpuFallback.meanAlongRows(a);
    }

    @Override
    public Tensor varianceAlongRows(Tensor a) {
        return cpuFallback.varianceAlongRows(a);
    }

    @Override
    public Tensor transpose(Tensor a) {
        return cpuFallback.transpose(a);
    }

    @Override
    public Tensor clamp(Tensor a, double min, double max) {
        return cpuFallback.clamp(a, min, max);
    }

    @Override
    public Tensor sqrt(Tensor a) {
        return cpuFallback.sqrt(a);
    }

    @Override
    public Tensor pow(Tensor a, double exponent) {
        return cpuFallback.pow(a, exponent);
    }

    @Override
    public double sum(Tensor a) {
        return cpuFallback.sum(a);
    }

    @Override
    public double sumAbs(Tensor a) {
        return cpuFallback.sumAbs(a);
    }

    @Override
    public Tensor multiplyScalar(Tensor a, double scalar) {
        return cpuFallback.multiplyScalar(a, scalar);
    }

    @Override
    public Tensor addScalar(Tensor a, double scalar) {
        return cpuFallback.addScalar(a, scalar);
    }

    @Override
    public Tensor divideScalar(Tensor a, double scalar) {
        return cpuFallback.divideScalar(a, scalar);
    }

    @Override
    public void print(Tensor t, String label) {
        cpuFallback.print(t, label);
    }
}