package io.github.kirstenali.deepj.tensor.cpu;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;

import java.util.Arrays;
import java.util.Random;

import static io.github.kirstenali.deepj.tensor.Tensor.requireSameShape;

abstract class CpuElementwiseOps implements TensorBackend {

    @FunctionalInterface
    interface FloatUnaryOp {
        float apply(float x);
    }

    @FunctionalInterface
    interface FloatBinaryOp {
        float apply(float x, float y);
    }

    static float fSqrt(float x) { return (float) Math.sqrt(x); }
    static float fPow(float x, float exponent) { return (float) Math.pow(x, exponent); }
    static float fExp(float x) { return (float) Math.exp(x); }
    static float fLog(float x) { return (float) Math.log(x); }
    static float fTanh(float x) { return (float) Math.tanh(x); }

    static void requireMatmulCompatible(Tensor a, Tensor b) {
        if (a.cols != b.rows) {
            throw new IllegalArgumentException(
                    "Shape mismatch for matmul: " + a.rows + "x" + a.cols +
                            " cannot be multiplied by " + b.rows + "x" + b.cols);
        }
    }

    static void requireRowVector(Tensor row, Tensor a) {
        if (row.rows != 1 || row.cols != a.cols)
            throw new IllegalArgumentException("rowVector must be 1x" + a.cols);
    }

    static void requireColVector(Tensor col, Tensor a) {
        if (col.cols != 1 || col.rows != a.rows)
            throw new IllegalArgumentException("colVector must be " + a.rows + "x1");
    }

    static void applyUnary(Tensor a, Tensor out, FloatUnaryOp fn) {
        DeepJExecutor.forRange(0, a.rows, r -> {
            int base = r * a.cols;
            for (int c = 0; c < a.cols; c++) {
                out.data[base + c] = fn.apply(a.data[base + c]);
            }
        });
    }

    static Tensor newUnary(Tensor a, FloatUnaryOp fn) {
        Tensor out = new Tensor(a.rows, a.cols);
        applyUnary(a, out, fn);
        return out;
    }

    static void applyBinary(Tensor a, Tensor b, Tensor out, FloatBinaryOp fn) {
        DeepJExecutor.forRange(0, a.rows, r -> {
            int base = r * a.cols;
            for (int c = 0; c < a.cols; c++) {
                out.data[base + c] = fn.apply(a.data[base + c], b.data[base + c]);
            }
        });
    }

    static Tensor newBinary(Tensor a, Tensor b, FloatBinaryOp fn) {
        Tensor out = new Tensor(a.rows, a.cols);
        applyBinary(a, b, out, fn);
        return out;
    }

    static void applyScalar(Tensor a, float s, Tensor out, FloatBinaryOp fn) {
        DeepJExecutor.forRange(0, a.rows, r -> {
            int base = r * a.cols;
            for (int c = 0; c < a.cols; c++) {
                out.data[base + c] = fn.apply(a.data[base + c], s);
            }
        });
    }

    static Tensor newScalar(Tensor a, float s, FloatBinaryOp fn) {
        Tensor out = new Tensor(a.rows, a.cols);
        applyScalar(a, s, out, fn);
        return out;
    }

    static void applyColBroadcast(Tensor a, Tensor col, Tensor out, FloatBinaryOp fn) {
        DeepJExecutor.forRange(0, a.rows, r -> {
            float v = col.data[r];
            int base = r * a.cols;
            for (int c = 0; c < a.cols; c++) {
                out.data[base + c] = fn.apply(a.data[base + c], v);
            }
        });
    }

    static Tensor newColBroadcast(Tensor a, Tensor col, FloatBinaryOp fn) {
        Tensor out = new Tensor(a.rows, a.cols);
        applyColBroadcast(a, col, out, fn);
        return out;
    }

    static void applyRowBroadcast(Tensor a, Tensor row, Tensor out, FloatBinaryOp fn) {
        DeepJExecutor.forRange(0, a.rows, r -> {
            int base = r * a.cols;
            for (int c = 0; c < a.cols; c++) {
                out.data[base + c] = fn.apply(a.data[base + c], row.data[c]);
            }
        });
    }

    static Tensor newRowBroadcast(Tensor a, Tensor row, FloatBinaryOp fn) {
        Tensor out = new Tensor(a.rows, a.cols);
        applyRowBroadcast(a, row, out, fn);
        return out;
    }

    public Tensor zeros(int rows, int cols) {
        return new Tensor(rows, cols);
    }

    public Tensor ones(int rows, int cols) {
        Tensor result = new Tensor(rows, cols);
        Arrays.fill(result.data, 1.0f);
        return result;
    }

    public Tensor random(int rows, int cols, Random rand) {
        Tensor t = new Tensor(rows, cols);
        for (int i = 0; i < t.data.length; i++) {
            t.data[i] = (float) (rand.nextGaussian() * 0.1);
        }
        return t;
    }

    public Tensor causalMask(int size) {
        Tensor mask = new Tensor(size, size);
        DeepJExecutor.forRange(0, size, r -> {
            int base = r * size;
            for (int c = 0; c < size; c++) {
                mask.data[base + c] = (c > r) ? -1e9f : 0.0f;
            }
        });
        return mask;
    }

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        requireMatmulCompatible(a, b);
        Tensor result = new Tensor(a.rows, b.cols);

        DeepJExecutor.forRange(0, a.rows, r -> {
            int aBase   = r * a.cols;
            int outBase = r * b.cols;
            for (int k = 0; k < a.cols; k++) {
                float aVal = a.data[aBase + k];
                int bBase = k * b.cols;
                for (int c = 0; c < b.cols; c++) {
                    result.data[outBase + c] += aVal * b.data[bBase + c];
                }
            }
        });

        return result;
    }

    @Override
    public Tensor add(Tensor a, Tensor b) {
        requireSameShape(a, b, "add");
        return newBinary(a, b, Float::sum);
    }
    @Override
    public Tensor subtract(Tensor a, Tensor b) {
        requireSameShape(a, b, "subtract");
        return newBinary(a, b, (x, y) -> x - y);
    }

    @Override
    public Tensor multiply(Tensor a, Tensor b) {
        requireSameShape(a, b, "multiply");
        return newBinary(a, b, (x, y) -> x * y);
    }

    @Override
    public Tensor divide(Tensor a, Tensor b) {
        requireSameShape(a, b, "divide");
        return newBinary(a, b, (x, y) -> x / y);
    }

    @Override
    public Tensor addRowVector(Tensor a, Tensor rv) {
        requireRowVector(rv, a);
        return newRowBroadcast(a, rv, Float::sum);
    }
    @Override public Tensor addBroadcastRows(Tensor a, Tensor rv)      { return addRowVector(a, rv); }
    @Override
    public Tensor multiplyBroadcastRows(Tensor a, Tensor rv) {
        requireRowVector(rv, a);
        return newRowBroadcast(a, rv, (x, y) -> x * y);
    }

    @Override
    public Tensor addBroadcastCols(Tensor a, Tensor cv) {
        requireColVector(cv, a);
        return newColBroadcast(a, cv, Float::sum);
    }
    @Override
    public Tensor subtractBroadcastCols(Tensor a, Tensor cv) {
        requireColVector(cv, a);
        return newColBroadcast(a, cv, (x, y) -> x - y);
    }

    @Override
    public Tensor multiplyBroadcastCols(Tensor a, Tensor cv) {
        requireColVector(cv, a);
        return newColBroadcast(a, cv, (x, y) -> x * y);
    }

    @Override
    public Tensor divideBroadcastCols(Tensor a, Tensor cv) {
        requireColVector(cv, a);
        return newColBroadcast(a, cv, (x, y) -> x / y);
    }

    @Override public Tensor multiplyScalar(Tensor a, float s) { return newScalar(a, s, (x, v) -> x * v); }
    @Override public Tensor addScalar(Tensor a, float s)      { return newScalar(a, s, Float::sum); }
    @Override public Tensor divideScalar(Tensor a, float s)   { return newScalar(a, s, (x, v) -> x / v); }


}
