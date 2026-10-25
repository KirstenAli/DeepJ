package io.github.kirstenali.deepj.tensor.cpu;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;

import java.util.Arrays;
import java.util.Random;

import static io.github.kirstenali.deepj.tensor.Tensor.requireSameShape;

abstract class CpuTrainingOps extends CpuReductionOps {
    @Override
    public float crossEntropyLoss(Tensor logits, int[] targets) {
        prepare(logits);
        Tensor.requireTargetsMatchRows(logits, targets);

        float lossSum = 0.0f;
        for (int i = 0; i < logits.rows; i++) {
            int base = i * logits.cols;
            int target = targets[i];

            float max = rowMax(logits.rawData(), base, logits.cols);
            float sumExp = rowSumExpShifted(logits.rawData(), base, logits.cols, max);

            lossSum += fLog(sumExp) + max - logits.rawData()[base + target];
        }
        return lossSum / logits.rows;
    }

    @Override
    public Tensor crossEntropyGradient(Tensor logits, int[] targets) {
        prepare(logits);
        Tensor.requireTargetsMatchRows(logits, targets);

        Tensor grad = new Tensor(logits.rows, logits.cols);

        for (int i = 0; i < logits.rows; i++) {
            int base = i * logits.cols;

            rowWriteSoftmax(logits.rawData(), grad.rawData(), base, logits.cols);
            grad.rawData()[base + targets[i]] -= 1.0f;
        }

        divideScalarInPlace(grad, logits.rows);
        return grad;
    }

    @Override
    public float crossEntropyLoss(Tensor logits, int[] targets, boolean[] mask) {
        prepare(logits);
        Tensor.requireTargetsMatchRows(logits, targets);
        float sum = 0.0f;
        for (int row = 0; row < logits.rows; row++) {
            if (mask[row]) sum += crossEntropyRowLoss(logits, targets[row], row);
        }
        return sum / includedRows(mask);
    }

    @Override
    public Tensor crossEntropyGradient(Tensor logits, int[] targets, boolean[] mask) {
        prepare(logits);
        Tensor.requireTargetsMatchRows(logits, targets);
        Tensor gradient = new Tensor(logits.rows, logits.cols);
        int included = includedRows(mask);
        for (int row = 0; row < logits.rows; row++) {
            if (mask[row]) writeCrossEntropyRow(gradient, logits, targets[row], row, included);
        }
        return gradient;
    }

    static float crossEntropyRowLoss(Tensor logits, int target, int row) {
        int base = row * logits.cols;
        float max = rowMax(logits.rawData(), base, logits.cols);
        return fLog(rowSumExpShifted(logits.rawData(), base, logits.cols, max))
                + max - logits.rawData()[base + target];
    }

    static void writeCrossEntropyRow(Tensor gradient, Tensor logits, int target,
                                             int row, int included) {
        int base = row * logits.cols;
        rowWriteSoftmax(logits.rawData(), gradient.rawData(), base, logits.cols);
        gradient.rawData()[base + target] -= 1.0f;
        for (int col = 0; col < logits.cols; col++) {
            gradient.rawData()[base + col] /= included;
        }
    }

    static int includedRows(boolean[] mask) {
        int count = 0;
        for (boolean included : mask) {
            if (included) count++;
        }
        return count;
    }

    @Override
    public void adamWUpdate(Tensor w, Tensor g, Tensor mt, Tensor vt,
                            float lr, float beta1, float beta2, float eps,
                            float weightDecay, float bc1, float bc2) {
        prepare(w, g, mt, vt);
        int n = w.rawData().length;
        for (int i = 0; i < n; i++) {
            float grad = g.rawData()[i];

            float mNew = beta1 * mt.rawData()[i] + (1.0f - beta1) * grad;
            float vNew = beta2 * vt.rawData()[i] + (1.0f - beta2) * (grad * grad);

            mt.rawData()[i] = mNew;
            vt.rawData()[i] = vNew;

            float mHat = mNew / bc1;
            float vHat = vNew / bc2;

            float update = (lr * mHat) / (fSqrt(vHat) + eps);

            if (weightDecay != 0.0f) update += lr * weightDecay * w.rawData()[i];

            w.rawData()[i] -= update;
        }
    }

    @Override
    public Tensor layerNormBackward(Tensor dXHat, Tensor xHat, Tensor std, int dim) {
        prepare(dXHat, xHat, std);
        Tensor dX = new Tensor(dXHat.rows, dXHat.cols);
        DeepJExecutor.forRange(0, dXHat.rows, row -> layerNormGradientRow(dX, dXHat, xHat, std, row, dim));
        return dX;
    }

    private static void layerNormGradientRow(Tensor output, Tensor gradient, Tensor normalized,
                                             Tensor std, int row, int width) {
        int base = row * width;
        float inverse = 1.0f / std.rawData()[row], sum = 0.0f, dot = 0.0f;
        for (int column = 0; column < width; column++) {
            float value = gradient.rawData()[base + column];
            sum += value;
            dot += value * normalized.rawData()[base + column];
        }
        for (int column = 0; column < width; column++) {
            float value = gradient.rawData()[base + column], norm = normalized.rawData()[base + column];
            output.rawData()[base + column] = inverse * (value - sum / width - norm * (dot / width));
        }
    }

    public float get(Tensor t, int r, int c)           { return t.data()[r * t.cols + c]; }
    public void   set(Tensor t, int r, int c, float v) { t.data()[r * t.cols + c] = v; }

    public Tensor getRow(Tensor t, int row) {
        Tensor result = new Tensor(1, t.cols);
        System.arraycopy(t.data(), row * t.cols, result.data(), 0, t.cols);
        return result;
    }

    public void setRow(Tensor t, int row, Tensor source, int srcRow) {
        System.arraycopy(source.data(), srcRow * source.cols, t.data(), row * t.cols, t.cols);
    }

    public Tensor sliceRows(Tensor t, int[] rowIndices, int cols) {
        Tensor out = new Tensor(rowIndices.length, cols);
        for (int i = 0; i < rowIndices.length; i++) {
            System.arraycopy(t.data(), rowIndices[i] * t.cols, out.data(), i * cols, cols);
        }
        return out;
    }

    @Override
    public void scatterAddRows(Tensor target, int[] indices, Tensor grad) {
        prepare(target, grad);
        for (int i = 0; i < indices.length; i++) {
            int tBase = indices[i] * target.cols;
            int gBase = i * target.cols;
            for (int c = 0; c < target.cols; c++) {
                target.rawData()[tBase + c] += grad.rawData()[gBase + c];
            }
        }
    }

    public Tensor sampleRows(Tensor t, int n, Random rnd) {
        Tensor out = new Tensor(n, t.cols);
        for (int i = 0; i < n; i++) {
            int r = rnd.nextInt(t.rows);
            System.arraycopy(t.data(), r * t.cols, out.data(), i * t.cols, t.cols);
        }
        return out;
    }

    @Override public void addInPlace(Tensor a, Tensor b)           { applyBinary(a, b, a, Float::sum); }
    @Override public void subtractInPlace(Tensor a, Tensor b)      { applyBinary(a, b, a, (x, y) -> x - y); }
    @Override public void multiplyInPlace(Tensor a, Tensor b)      { applyBinary(a, b, a, (x, y) -> x * y); }
    @Override public void divideInPlace(Tensor a, Tensor b)        { applyBinary(a, b, a, (x, y) -> x / y); }

    @Override public void multiplyScalarInPlace(Tensor a, float s) { applyScalar(a, s, a, (x, v) -> x * v); }
    @Override public void addScalarInPlace(Tensor a, float s)      { applyScalar(a, s, a, Float::sum); }
    @Override public void divideScalarInPlace(Tensor a, float s)   { applyScalar(a, s, a, (x, v) -> x / v); }

    @Override public void sqrtInPlace(Tensor a)    { applyUnary(a, a, CpuBackend::fSqrt); }
    @Override public void negInPlace(Tensor a)     { applyUnary(a, a, x -> -x); }
    @Override public void expInPlace(Tensor a)     { applyUnary(a, a, CpuBackend::fExp); }
    @Override public void logInPlace(Tensor a)     { applyUnary(a, a, CpuBackend::fLog); }
    @Override public void reluInPlace(Tensor a)    { applyUnary(a, a, x -> Math.max(0.0f, x)); }
    @Override public void geluInPlace(Tensor a)    { applyUnary(a, a, CpuBackend::geluScalar); }
    @Override public void tanhInPlace(Tensor a)    { applyUnary(a, a, CpuBackend::fTanh); }
    @Override public void sigmoidInPlace(Tensor a) { applyUnary(a, a, x -> 1.0f / (1.0f + fExp(-x))); }

    public void print(Tensor t, String label) {
        System.out.println(label);
        for (int r = 0; r < t.rows; r++) {
            int base = r * t.cols;
            for (int c = 0; c < t.cols; c++) {
                System.out.printf("%.4f ", t.data()[base + c]);
            }
            System.out.println();
        }
    }
}
