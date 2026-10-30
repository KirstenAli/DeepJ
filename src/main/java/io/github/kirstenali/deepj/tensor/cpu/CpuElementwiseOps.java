package io.github.kirstenali.deepj.tensor.cpu;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.CpuNdElementwise;
import io.github.kirstenali.deepj.tensor.CpuNdMatmul;

import java.util.Arrays;
import java.util.Random;

import static io.github.kirstenali.deepj.tensor.Tensor.requireSameShape;

abstract class CpuElementwiseOps implements TensorBackend {

    static void prepare(Tensor... tensors) {
        for (Tensor tensor : tensors) {
            if (tensor.rawData() == null || tensor.getGpuTag() != null) tensor.materialize();
        }
    }

    @FunctionalInterface
    interface FloatUnaryOp extends CpuNdElementwise.Unary {}

    @FunctionalInterface
    interface FloatBinaryOp extends CpuNdElementwise.Binary {}

    static float fSqrt(float x) { return (float) Math.sqrt(x); }
    static float fPow(float x, float exponent) { return (float) Math.pow(x, exponent); }
    static float fExp(float x) { return (float) Math.exp(x); }
    static float fLog(float x) { return (float) Math.log(x); }
    static float fTanh(float x) { return (float) Math.tanh(x); }

    static void requireMatmulCompatible(Tensor a, Tensor b) {
        if (a.matrixCols() != b.matrixRows()) {
            throw new IllegalArgumentException(
                    "Shape mismatch for matmul: " + a.matrixRows() + "x" + a.matrixCols() +
                            " cannot be multiplied by " + b.matrixRows() + "x" + b.matrixCols());
        }
    }

    static void requireRowVector(Tensor row, Tensor a) {
        if (row.matrixRows() != 1 || row.matrixCols() != a.matrixCols())
            throw new IllegalArgumentException("rowVector must be 1x" + a.matrixCols());
    }

    static void requireColVector(Tensor col, Tensor a) {
        if (col.matrixCols() != 1 || col.matrixRows() != a.matrixRows())
            throw new IllegalArgumentException("colVector must be " + a.matrixRows() + "x1");
    }

    static void applyUnary(Tensor a, Tensor out, FloatUnaryOp fn) {
        if (!CpuNdElementwise.sameMatrices(a, out)) {
            CpuNdElementwise.unaryInto(a, out, fn);
            return;
        }
        prepare(a, out);
        float[] input = a.rawData(), output = out.rawData();
        int columns = a.matrixCols();
        DeepJExecutor.forRange(0, a.matrixRows(), r -> {
            int base = r * columns;
            for (int c = 0; c < columns; c++) {
                output[base + c] = fn.apply(input[base + c]);
            }
        });
    }

    static Tensor newUnary(Tensor a, FloatUnaryOp fn) {
        if (!CpuNdElementwise.matrix(a)) return CpuNdElementwise.unary(a, fn);
        Tensor out = new Tensor(a.matrixRows(), a.matrixCols());
        applyUnary(a, out, fn);
        return out;
    }

    static void applyBinary(Tensor a, Tensor b, Tensor out, FloatBinaryOp fn) {
        if (!CpuNdElementwise.sameMatrices(a, b) || !CpuNdElementwise.sameMatrices(a, out)) {
            CpuNdElementwise.binaryInto(a, b, out, fn);
            return;
        }
        prepare(a, b, out);
        float[] left = a.rawData(), right = b.rawData(), output = out.rawData();
        int columns = a.matrixCols();
        DeepJExecutor.forRange(0, a.matrixRows(), r -> {
            int base = r * columns;
            for (int c = 0; c < columns; c++) {
                output[base + c] = fn.apply(left[base + c], right[base + c]);
            }
        });
    }

    static Tensor newBinary(Tensor a, Tensor b, FloatBinaryOp fn) {
        if (!CpuNdElementwise.sameMatrices(a, b)) return CpuNdElementwise.binary(a, b, fn);
        Tensor out = new Tensor(a.matrixRows(), a.matrixCols());
        applyBinary(a, b, out, fn);
        return out;
    }

    static void applyScalar(Tensor a, float s, Tensor out, FloatBinaryOp fn) {
        if (!CpuNdElementwise.sameMatrices(a, out)) {
            CpuNdElementwise.unaryInto(a, out, value -> fn.apply(value, s));
            return;
        }
        prepare(a, out);
        float[] input = a.rawData(), output = out.rawData();
        int columns = a.matrixCols();
        DeepJExecutor.forRange(0, a.matrixRows(), r -> {
            int base = r * columns;
            for (int c = 0; c < columns; c++) {
                output[base + c] = fn.apply(input[base + c], s);
            }
        });
    }

    static Tensor newScalar(Tensor a, float s, FloatBinaryOp fn) {
        if (!CpuNdElementwise.matrix(a)) return CpuNdElementwise.unary(a, value -> fn.apply(value, s));
        Tensor out = new Tensor(a.matrixRows(), a.matrixCols());
        applyScalar(a, s, out, fn);
        return out;
    }

    static void applyColBroadcast(Tensor a, Tensor col, Tensor out, FloatBinaryOp fn) {
        prepare(a, col, out);
        float[] input = a.rawData(), column = col.rawData(), output = out.rawData();
        int columns = a.matrixCols();
        DeepJExecutor.forRange(0, a.matrixRows(), r -> {
            float v = column[r];
            int base = r * columns;
            for (int c = 0; c < columns; c++) {
                output[base + c] = fn.apply(input[base + c], v);
            }
        });
    }

    static Tensor newColBroadcast(Tensor a, Tensor col, FloatBinaryOp fn) {
        if (!CpuNdElementwise.matrix(a) || !CpuNdElementwise.matrix(col)) {
            return CpuNdElementwise.binary(a, col, fn);
        }
        Tensor out = new Tensor(a.matrixRows(), a.matrixCols());
        applyColBroadcast(a, col, out, fn);
        return out;
    }

    static void applyRowBroadcast(Tensor a, Tensor row, Tensor out, FloatBinaryOp fn) {
        prepare(a, row, out);
        float[] input = a.rawData(), vector = row.rawData(), output = out.rawData();
        int columns = a.matrixCols();
        DeepJExecutor.forRange(0, a.matrixRows(), r -> {
            int base = r * columns;
            for (int c = 0; c < columns; c++) {
                output[base + c] = fn.apply(input[base + c], vector[c]);
            }
        });
    }

    static Tensor newRowBroadcast(Tensor a, Tensor row, FloatBinaryOp fn) {
        if (!CpuNdElementwise.matrix(a) || !CpuNdElementwise.matrix(row)) {
            return CpuNdElementwise.binary(a, row, fn);
        }
        Tensor out = new Tensor(a.matrixRows(), a.matrixCols());
        applyRowBroadcast(a, row, out, fn);
        return out;
    }

    public Tensor zeros(int rows, int cols) {
        return new Tensor(rows, cols);
    }

    public Tensor ones(int rows, int cols) {
        Tensor result = new Tensor(rows, cols);
        Arrays.fill(result.rawData(), 1.0f);
        return result;
    }

    public Tensor random(int rows, int cols, Random rand) {
        Tensor t = new Tensor(rows, cols);
        fillRandom(t.rawData(), rand);
        return t;
    }

    private static void fillRandom(float[] values, Random rand) {
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) (rand.nextGaussian() * 0.1);
        }
    }

    public Tensor causalMask(int size) {
        Tensor mask = new Tensor(size, size);
        DeepJExecutor.forRange(0, size, r -> {
            int base = r * size;
            for (int c = 0; c < size; c++) {
                mask.rawData()[base + c] = (c > r) ? -1e9f : 0.0f;
            }
        });
        return mask;
    }

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        if (!CpuNdElementwise.matrix(a) || !CpuNdElementwise.matrix(b)) return CpuNdMatmul.apply(a, b);
        prepare(a, b);
        requireMatmulCompatible(a, b);
        Tensor result = new Tensor(a.matrixRows(), b.matrixCols());
        float[] left = a.rawData(), right = b.rawData(), output = result.rawData();
        DeepJExecutor.forRange(0, a.matrixRows(),
                row -> matmulRow(left, right, output, row, a.matrixCols(), b.matrixCols()));
        return result;
    }

    private static void matmulRow(float[] left, float[] right, float[] output,
                                   int row, int inner, int columns) {
        int aBase = row * inner, outBase = row * columns;
        for (int k = 0; k < inner; k++) {
            float value = left[aBase + k];
            int bBase = k * columns;
            for (int column = 0; column < columns; column++) {
                output[outBase + column] += value * right[bBase + column];
            }
        }
    }

    @Override
    public Tensor add(Tensor a, Tensor b) {
        return newBinary(a, b, Float::sum);
    }
    @Override
    public Tensor subtract(Tensor a, Tensor b) {
        return newBinary(a, b, (x, y) -> x - y);
    }

    @Override
    public Tensor multiply(Tensor a, Tensor b) {
        return newBinary(a, b, (x, y) -> x * y);
    }

    @Override
    public Tensor divide(Tensor a, Tensor b) {
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
