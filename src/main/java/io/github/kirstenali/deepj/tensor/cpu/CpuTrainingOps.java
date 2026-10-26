package io.github.kirstenali.deepj.tensor.cpu;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.CpuNdIndexing;
import io.github.kirstenali.deepj.tensor.CpuNdCrossEntropy;
import io.github.kirstenali.deepj.tensor.CpuNdElementwise;
import io.github.kirstenali.deepj.tensor.CrossEntropyChecks;
import io.github.kirstenali.deepj.tensor.CpuNdAdamW;
import io.github.kirstenali.deepj.tensor.OptimizerChecks;
import io.github.kirstenali.deepj.tensor.NormalizationChecks;
import io.github.kirstenali.deepj.tensor.CpuNdNormalization;
import io.github.kirstenali.deepj.tensor.CpuRowAccess;

import java.util.Arrays;
import java.util.Random;

import static io.github.kirstenali.deepj.tensor.Tensor.requireSameShape;

abstract class CpuTrainingOps extends CpuReductionOps {
    @Override
    public float crossEntropyLoss(Tensor logits, int[] targets) {
        if (!CpuNdElementwise.matrix(logits)) return CpuNdCrossEntropy.loss(logits, targets, null);
        prepare(logits);
        Tensor.requireTargetsMatchRows(logits, targets);

        float lossSum = 0.0f;
        for (int i = 0; i < logits.matrixRows(); i++) {
            int base = i * logits.matrixCols();
            int target = targets[i];

            float max = rowMax(logits.rawData(), base, logits.matrixCols());
            float sumExp = rowSumExpShifted(logits.rawData(), base, logits.matrixCols(), max);

            lossSum += fLog(sumExp) + max - logits.rawData()[base + target];
        }
        return lossSum / logits.matrixRows();
    }

    @Override
    public Tensor crossEntropyGradient(Tensor logits, int[] targets) {
        if (!CpuNdElementwise.matrix(logits)) return CpuNdCrossEntropy.apply(logits, targets, null).gradient();
        prepare(logits);
        Tensor.requireTargetsMatchRows(logits, targets);

        Tensor grad = new Tensor(logits.matrixRows(), logits.matrixCols());

        for (int i = 0; i < logits.matrixRows(); i++) {
            int base = i * logits.matrixCols();

            rowWriteSoftmax(logits.rawData(), grad.rawData(), base, logits.matrixCols());
            grad.rawData()[base + targets[i]] -= 1.0f;
        }

        divideScalarInPlace(grad, logits.matrixRows());
        return grad;
    }

    @Override
    public float crossEntropyLoss(Tensor logits, int[] targets, boolean[] mask) {
        CrossEntropyChecks.included(mask, CrossEntropyChecks.positions(logits));
        if (!CpuNdElementwise.matrix(logits)) return CpuNdCrossEntropy.loss(logits, targets, mask);
        prepare(logits);
        Tensor.requireTargetsMatchRows(logits, targets);
        float sum = 0.0f;
        for (int row = 0; row < logits.matrixRows(); row++) {
            if (mask[row]) sum += crossEntropyRowLoss(logits, targets[row], row);
        }
        return sum / includedRows(mask);
    }

    @Override
    public Tensor crossEntropyGradient(Tensor logits, int[] targets, boolean[] mask) {
        CrossEntropyChecks.included(mask, CrossEntropyChecks.positions(logits));
        if (!CpuNdElementwise.matrix(logits)) return CpuNdCrossEntropy.apply(logits, targets, mask).gradient();
        prepare(logits);
        Tensor.requireTargetsMatchRows(logits, targets);
        Tensor gradient = new Tensor(logits.matrixRows(), logits.matrixCols());
        int included = includedRows(mask);
        for (int row = 0; row < logits.matrixRows(); row++) {
            if (mask[row]) writeCrossEntropyRow(gradient, logits, targets[row], row, included);
        }
        return gradient;
    }

    static float crossEntropyRowLoss(Tensor logits, int target, int row) {
        int base = row * logits.matrixCols();
        float max = rowMax(logits.rawData(), base, logits.matrixCols());
        return fLog(rowSumExpShifted(logits.rawData(), base, logits.matrixCols(), max))
                + max - logits.rawData()[base + target];
    }

    static void writeCrossEntropyRow(Tensor gradient, Tensor logits, int target,
                                             int row, int included) {
        int base = row * logits.matrixCols();
        rowWriteSoftmax(logits.rawData(), gradient.rawData(), base, logits.matrixCols());
        gradient.rawData()[base + target] -= 1.0f;
        for (int col = 0; col < logits.matrixCols(); col++) {
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
        OptimizerChecks.requireState(w, g, mt, vt);
        if (!CpuNdElementwise.sameMatrices(w, g) || !CpuNdElementwise.sameMatrices(mt, vt)) {
            CpuNdAdamW.update(w, g, mt, vt, lr, beta1, beta2, eps, weightDecay, bc1, bc2);
            return;
        }
        adamMatrixUpdate(w, g, mt, vt, lr, beta1, beta2, eps, weightDecay, bc1, bc2);
    }

    private static void adamMatrixUpdate(Tensor w, Tensor g, Tensor mt, Tensor vt,
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
        Tensor.requireSameShape(dXHat, xHat, "layerNormBackward");
        if (dim != NormalizationChecks.width(xHat)) throw new IllegalArgumentException("LayerNorm width mismatch");
        NormalizationChecks.statistics(xHat, std);
        if (!CpuNdElementwise.sameMatrices(dXHat, xHat) || !CpuNdElementwise.matrix(std)) {
            return CpuNdNormalization.backward(dXHat, xHat, std, Tensor.ones(1, dim), true);
        }
        prepare(dXHat, xHat, std);
        Tensor dX = new Tensor(dXHat.matrixRows(), dXHat.matrixCols());
        DeepJExecutor.forRange(0, dXHat.matrixRows(), row -> layerNormGradientRow(dX, dXHat, xHat, std, row, dim));
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

    public float get(Tensor t, int r, int c)           { return t.get(r, c); }
    public void   set(Tensor t, int r, int c, float v) { t.set(r, c, v); }

    public Tensor getRow(Tensor t, int row) {
        return CpuRowAccess.select(t, new int[]{row}, t.matrixCols());
    }

    public void setRow(Tensor t, int row, Tensor source, int srcRow) {
        CpuRowAccess.assign(t, row, source, srcRow);
    }

    public Tensor sliceRows(Tensor t, int[] rowIndices, int cols) {
        return CpuRowAccess.select(t, rowIndices, cols);
    }

    @Override
    public void scatterAddRows(Tensor target, int[] indices, Tensor grad) {
        if (target.rank() != 2 || grad.rank() != 2) throw new IllegalArgumentException("Row scatter requires matrices");
        CpuNdIndexing.add(target, 0, indices, grad);
    }

    public Tensor sampleRows(Tensor t, int n, Random rnd) {
        return CpuRowAccess.sample(t, n, rnd);
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
        if (t.rank() != 2) {
            System.out.println(t.dtype() + " shape=" + java.util.Arrays.toString(t.shape())
                    + " values=" + java.util.Arrays.toString(t.toFloatArray()));
            return;
        }
        for (int r = 0; r < t.matrixRows(); r++) {
            for (float value : t.rowData(r)) {
                System.out.printf("%.4f ", value);
            }
            System.out.println();
        }
    }
}
