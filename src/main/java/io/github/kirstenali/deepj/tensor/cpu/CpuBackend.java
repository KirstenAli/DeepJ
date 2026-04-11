package io.github.kirstenali.deepj.tensor.cpu;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;

import java.util.Arrays;
import java.util.Random;

import static io.github.kirstenali.deepj.tensor.Tensor.requireSameShape;

public final class CpuBackend implements TensorBackend {

    @FunctionalInterface
    private interface FloatUnaryOp {
        float apply(float x);
    }

    @FunctionalInterface
    private interface FloatBinaryOp {
        float apply(float x, float y);
    }

    private static float fSqrt(float x) { return (float) Math.sqrt(x); }
    private static float fPow(float x, float exponent) { return (float) Math.pow(x, exponent); }
    private static float fExp(float x) { return (float) Math.exp(x); }
    private static float fLog(float x) { return (float) Math.log(x); }
    private static float fTanh(float x) { return (float) Math.tanh(x); }

    // ══════════════════════════════════════════════════════════════════
    //  Validation
    // ══════════════════════════════════════════════════════════════════

    private static void requireMatmulCompatible(Tensor a, Tensor b) {
        if (a.cols != b.rows) {
            throw new IllegalArgumentException(
                    "Shape mismatch for matmul: " + a.rows + "x" + a.cols +
                            " cannot be multiplied by " + b.rows + "x" + b.cols);
        }
    }

    private static void requireRowVector(Tensor row, Tensor a) {
        if (row.rows != 1 || row.cols != a.cols)
            throw new IllegalArgumentException("rowVector must be 1x" + a.cols);
    }

    private static void requireColVector(Tensor col, Tensor a) {
        if (col.cols != 1 || col.rows != a.rows)
            throw new IllegalArgumentException("colVector must be " + a.rows + "x1");
    }

    // ══════════════════════════════════════════════════════════════════
    //  Core element-wise helpers (flat row-major)
    // ══════════════════════════════════════════════════════════════════

    // ── unary: out[r*cols+c] = fn(a[r*cols+c]) ─────────────────────

    static void applyUnary(Tensor a, Tensor out, FloatUnaryOp fn) {
        DeepJExecutor.forRange(0, a.rows, r -> {
            int base = r * a.cols;
            for (int c = 0; c < a.cols; c++) out.data[base + c] = fn.apply(a.data[base + c]);
        });
    }

    private static Tensor newUnary(Tensor a, FloatUnaryOp fn) {
        Tensor out = new Tensor(a.rows, a.cols);
        applyUnary(a, out, fn);
        return out;
    }

    // ── binary: out[i] = fn(a[i], b[i]) ───────────────────────────

    static void applyBinary(Tensor a, Tensor b, Tensor out, FloatBinaryOp fn) {
        DeepJExecutor.forRange(0, a.rows, r -> {
            int base = r * a.cols;
            for (int c = 0; c < a.cols; c++) out.data[base + c] = fn.apply(a.data[base + c], b.data[base + c]);
        });
    }

    private static Tensor newBinary(Tensor a, Tensor b, FloatBinaryOp fn) {
        Tensor out = new Tensor(a.rows, a.cols);
        applyBinary(a, b, out, fn);
        return out;
    }

    // ── scalar: out[i] = fn(a[i], scalar) ────────────────────────

    static void applyScalar(Tensor a, float s, Tensor out, FloatBinaryOp fn) {
        DeepJExecutor.forRange(0, a.rows, r -> {
            int base = r * a.cols;
            for (int c = 0; c < a.cols; c++) out.data[base + c] = fn.apply(a.data[base + c], s);
        });
    }

    private static Tensor newScalar(Tensor a, float s, FloatBinaryOp fn) {
        Tensor out = new Tensor(a.rows, a.cols);
        applyScalar(a, s, out, fn);
        return out;
    }

    // ── column broadcast: out[r,c] = fn(a[r,c], col[r,0]) ─────────
    //    col is rows×1, so col.data[r] is element (r,0)

    static void applyColBroadcast(Tensor a, Tensor col, Tensor out, FloatBinaryOp fn) {
        DeepJExecutor.forRange(0, a.rows, r -> {
            float v = col.data[r]; // col is Rx1 → data[r*1+0] = data[r]
            int base = r * a.cols;
            for (int c = 0; c < a.cols; c++) out.data[base + c] = fn.apply(a.data[base + c], v);
        });
    }

    private static Tensor newColBroadcast(Tensor a, Tensor col, FloatBinaryOp fn) {
        Tensor out = new Tensor(a.rows, a.cols);
        applyColBroadcast(a, col, out, fn);
        return out;
    }

    // ── row broadcast: out[r,c] = fn(a[r,c], row[0,c]) ────────────
    //    row is 1×cols, so row.data[c] is element (0,c)

    static void applyRowBroadcast(Tensor a, Tensor row, Tensor out, FloatBinaryOp fn) {
        DeepJExecutor.forRange(0, a.rows, r -> {
            int base = r * a.cols;
            for (int c = 0; c < a.cols; c++) out.data[base + c] = fn.apply(a.data[base + c], row.data[c]);
        });
    }

    private static Tensor newRowBroadcast(Tensor a, Tensor row, FloatBinaryOp fn) {
        Tensor out = new Tensor(a.rows, a.cols);
        applyRowBroadcast(a, row, out, fn);
        return out;
    }

    // ══════════════════════════════════════════════════════════════════
    //  Factories
    // ══════════════════════════════════════════════════════════════════

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
        for (int i = 0; i < t.data.length; i++) t.data[i] = (float) (rand.nextGaussian() * 0.1);
        return t;
    }

    public Tensor causalMask(int size) {
        Tensor mask = new Tensor(size, size);
        DeepJExecutor.forRange(0, size, r -> {
            int base = r * size;
            for (int c = 0; c < size; c++) mask.data[base + c] = (c > r) ? -1e9f : 0.0f;
        });
        return mask;
    }


    // ══════════════════════════════════════════════════════════════════
    //  Core binary ops
    // ══════════════════════════════════════════════════════════════════

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        requireMatmulCompatible(a, b);
        Tensor result = new Tensor(a.rows, b.cols);

        // ikj loop order: cache-friendly for flat row-major arrays.
        // Inner loop strides over consecutive b.data elements (row k of B).
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

    @Override public Tensor add(Tensor a, Tensor b)      { requireSameShape(a, b, "add");      return newBinary(a, b, Float::sum); }
    @Override public Tensor subtract(Tensor a, Tensor b)  { requireSameShape(a, b, "subtract"); return newBinary(a, b, (x, y) -> x - y); }
    @Override public Tensor multiply(Tensor a, Tensor b)  { requireSameShape(a, b, "multiply"); return newBinary(a, b, (x, y) -> x * y); }
    @Override public Tensor divide(Tensor a, Tensor b)    { requireSameShape(a, b, "divide");   return newBinary(a, b, (x, y) -> x / y); }

    // ══════════════════════════════════════════════════════════════════
    //  Broadcasts
    // ══════════════════════════════════════════════════════════════════

    @Override public Tensor addRowVector(Tensor a, Tensor rv)          { requireRowVector(rv, a); return newRowBroadcast(a, rv, Float::sum); }
    @Override public Tensor addBroadcastRows(Tensor a, Tensor rv)      { return addRowVector(a, rv); }
    @Override public Tensor multiplyBroadcastRows(Tensor a, Tensor rv) { requireRowVector(rv, a); return newRowBroadcast(a, rv, (x, y) -> x * y); }

    @Override public Tensor addBroadcastCols(Tensor a, Tensor cv)      { requireColVector(cv, a); return newColBroadcast(a, cv, Float::sum); }
    @Override public Tensor subtractBroadcastCols(Tensor a, Tensor cv)  { requireColVector(cv, a); return newColBroadcast(a, cv, (x, y) -> x - y); }
    @Override public Tensor multiplyBroadcastCols(Tensor a, Tensor cv)  { requireColVector(cv, a); return newColBroadcast(a, cv, (x, y) -> x * y); }
    @Override public Tensor divideBroadcastCols(Tensor a, Tensor cv)    { requireColVector(cv, a); return newColBroadcast(a, cv, (x, y) -> x / y); }

    // ══════════════════════════════════════════════════════════════════
    //  Scalar ops
    // ══════════════════════════════════════════════════════════════════

    @Override public Tensor multiplyScalar(Tensor a, float s) { return newScalar(a, s, (x, v) -> x * v); }
    @Override public Tensor addScalar(Tensor a, float s)      { return newScalar(a, s, Float::sum); }
    @Override public Tensor divideScalar(Tensor a, float s)   { return newScalar(a, s, (x, v) -> x / v); }

    // ══════════════════════════════════════════════════════════════════
    //  Reductions / statistics
    // ══════════════════════════════════════════════════════════════════

    @Override
    public Tensor sumRows(Tensor a) {
        Tensor result = new Tensor(1, a.cols);
        for (int r = 0; r < a.rows; r++) {
            int base = r * a.cols;
            for (int c = 0; c < a.cols; c++) result.data[c] += a.data[base + c];
        }
        return result;
    }

    @Override
    public Tensor sumAlongRows(Tensor a) {
        Tensor result = new Tensor(a.rows, 1);
        DeepJExecutor.forRange(0, a.rows, r -> {
            int base = r * a.cols;
            result.data[r] = rowSum(a.data, base, a.cols);
        });
        return result;
    }

    @Override public Tensor sumAlongCols(Tensor a) { return sumRows(a); }

    @Override
    public Tensor meanAlongRows(Tensor a) {
        Tensor result = new Tensor(a.rows, 1);
        float invCols = 1.0f / a.cols;
        DeepJExecutor.forRange(0, a.rows, r -> {
            int base = r * a.cols;
            result.data[r] = rowSum(a.data, base, a.cols) * invCols;
        });
        return result;
    }

    @Override
    public Tensor varianceAlongRows(Tensor a) {
        Tensor result = new Tensor(a.rows, 1);
        float invCols = 1.0f / a.cols;
        DeepJExecutor.forRange(0, a.rows, r -> {
            int base = r * a.cols;
            float mean = rowSum(a.data, base, a.cols) * invCols;
            float acc = 0.0f;
            for (int c = 0; c < a.cols; c++) {
                float diff = a.data[base + c] - mean;
                acc += diff * diff;
            }
            result.data[r] = acc * invCols;
        });
        return result;
    }

    @Override
    public Tensor maxAlongRows(Tensor a) {
        Tensor result = new Tensor(a.rows, 1);
        DeepJExecutor.forRange(0, a.rows, r -> {
            int base = r * a.cols;
            float max = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < a.cols; c++) if (a.data[base + c] > max) max = a.data[base + c];
            result.data[r] = max;
        });
        return result;
    }

    public float sum(Tensor a) {
        float s = 0.0f;
        for (float v : a.data) s += v;
        return s;
    }

    public float sumAbs(Tensor a) {
        float s = 0.0f;
        for (float v : a.data) s += Math.abs(v);
        return s;
    }

    // ══════════════════════════════════════════════════════════════════
    //  Unary math
    // ══════════════════════════════════════════════════════════════════

    @Override
    public Tensor transpose(Tensor a) {
        Tensor result = new Tensor(a.cols, a.rows);
        DeepJExecutor.forRange(0, a.rows, r -> {
            int aBase = r * a.cols;
            for (int c = 0; c < a.cols; c++) result.data[c * a.rows + r] = a.data[aBase + c];
        });
        return result;
    }

    @Override public Tensor clamp(Tensor a, float min, float max) { return newUnary(a, x -> Math.min(max, Math.max(min, x))); }
    @Override public Tensor sqrt(Tensor a)                          { return newUnary(a, CpuBackend::fSqrt); }
    @Override public Tensor pow(Tensor a, float exponent)                    { return newUnary(a, x -> fPow(x, exponent)); }
    @Override public Tensor neg(Tensor a)                           { return newUnary(a, x -> -x); }
    @Override public Tensor exp(Tensor a)                           { return newUnary(a, CpuBackend::fExp); }
    @Override public Tensor log(Tensor a)                           { return newUnary(a, CpuBackend::fLog); }

    // ══════════════════════════════════════════════════════════════════
    //  Activations
    // ══════════════════════════════════════════════════════════════════

    @Override public Tensor tanh(Tensor a)    { return newUnary(a, CpuBackend::fTanh); }
    @Override public Tensor sigmoid(Tensor a) { return newUnary(a, x -> 1.0f / (1.0f + fExp(-x))); }
    @Override public Tensor relu(Tensor a)    { return newUnary(a, x -> Math.max(0.0f, x)); }
    @Override public Tensor gelu(Tensor a)    { return newUnary(a, CpuBackend::geluScalar); }

    @Override
    public Tensor reluBackward(Tensor input, Tensor gradOutput) {
        requireSameShape(input, gradOutput, "reluBackward");
        return newBinary(input, gradOutput, (x, g) -> x > 0.0f ? g : 0.0f);
    }

    @Override
    public Tensor geluBackward(Tensor input, Tensor gradOutput) {
        requireSameShape(input, gradOutput, "geluBackward");
        return newBinary(input, gradOutput, (x, g) -> g * geluDerivScalar(x));
    }

    private static float geluScalar(float x) {
        float c = fSqrt(2.0f / (float) Math.PI);
        float x3 = x * x * x;
        float t = c * (x + 0.044715f * x3);
        return 0.5f * x * (1.0f + fTanh(t));
    }

    private static float geluDerivScalar(float x) {
        float c = fSqrt(2.0f / (float) Math.PI);
        float x2 = x * x;
        float x3 = x2 * x;
        float t = c * (x + 0.044715f * x3);
        float tanhT = fTanh(t);
        float sech2 = 1.0f - tanhT * tanhT;
        float dtDx = c * (1.0f + 3.0f * 0.044715f * x2);
        return 0.5f * (1.0f + tanhT) + 0.5f * x * sech2 * dtDx;
    }

    // ══════════════════════════════════════════════════════════════════
    //  Softmax / cross-entropy internals
    // ══════════════════════════════════════════════════════════════════

    private static float rowMax(float[] data, int base, int cols) {
        float max = Float.NEGATIVE_INFINITY;
        for (int c = 0; c < cols; c++) if (data[base + c] > max) max = data[base + c];
        return max;
    }

    private static float rowSum(float[] data, int base, int cols) {
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) sum += data[base + c];
        return sum;
    }

    private static float rowDot(float[] a, float[] b, int base, int cols) {
        float dot = 0.0f;
        for (int c = 0; c < cols; c++) dot += a[base + c] * b[base + c];
        return dot;
    }

    private static float rowSumExpShifted(float[] data, int base, int cols, float max) {
        float sumExp = 0.0f;
        for (int c = 0; c < cols; c++) sumExp += fExp(data[base + c] - max);
        return sumExp;
    }

    private static void rowWriteExpShifted(float[] src, float[] out, int base, int cols, float max) {
        for (int c = 0; c < cols; c++) out[base + c] = fExp(src[base + c] - max);
    }

    private static void rowNormalizeInPlace(float[] data, int base, int cols, float denom) {
        for (int c = 0; c < cols; c++) data[base + c] /= denom;
    }

    private static void rowWriteSoftmax(float[] logits, float[] out, int base, int cols) {
        float max = rowMax(logits, base, cols);
        rowWriteExpShifted(logits, out, base, cols, max);
        float sumExp = rowSumExpShifted(logits, base, cols, max);
        rowNormalizeInPlace(out, base, cols, sumExp);
    }

    @Override
    public Tensor softmaxRows(Tensor logits) {
        Tensor result = new Tensor(logits.rows, logits.cols);

        DeepJExecutor.forRange(0, logits.rows, r -> {
            int base = r * logits.cols;
            rowWriteSoftmax(logits.data, result.data, base, logits.cols);
        });

        return result;
    }

    @Override
    public Tensor softmaxBackward(Tensor gradOutput, Tensor softmaxOut) {
        requireSameShape(gradOutput, softmaxOut, "softmaxBackward");
        Tensor result = new Tensor(gradOutput.rows, gradOutput.cols);

        DeepJExecutor.forRange(0, gradOutput.rows, r -> {
            int base = r * gradOutput.cols;

            float dot = rowDot(gradOutput.data, softmaxOut.data, base, gradOutput.cols);

            for (int c = 0; c < gradOutput.cols; c++)
                result.data[base + c] = softmaxOut.data[base + c] * (gradOutput.data[base + c] - dot);
        });

        return result;
    }

    // ══════════════════════════════════════════════════════════════════
    //  Fused high-level ops
    // ══════════════════════════════════════════════════════════════════

    public float crossEntropyLoss(Tensor logits, int[] targets) {
        Tensor.requireTargetsMatchRows(logits, targets);

        float lossSum = 0.0f;
        for (int i = 0; i < logits.rows; i++) {
            int base = i * logits.cols;
            int target = targets[i];

            float max = rowMax(logits.data, base, logits.cols);
            float sumExp = rowSumExpShifted(logits.data, base, logits.cols, max);

            lossSum += fLog(sumExp) + max - logits.data[base + target];
        }
        return lossSum / logits.rows;
    }

    @Override
    public Tensor crossEntropyGradient(Tensor logits, int[] targets) {
        Tensor.requireTargetsMatchRows(logits, targets);

        Tensor grad = new Tensor(logits.rows, logits.cols);

        for (int i = 0; i < logits.rows; i++) {
            int base = i * logits.cols;

            rowWriteSoftmax(logits.data, grad.data, base, logits.cols);
            grad.data[base + targets[i]] -= 1.0f;
        }

        divideScalarInPlace(grad, logits.rows);
        return grad;
    }

    @Override
    public void adamWUpdate(Tensor w, Tensor g, Tensor mt, Tensor vt,
                            float lr, float beta1, float beta2, float eps,
                            float weightDecay, float bc1, float bc2) {
        int n = w.data.length;
        for (int i = 0; i < n; i++) {
            float grad = g.data[i];

            float mNew = beta1 * mt.data[i] + (1.0f - beta1) * grad;
            float vNew = beta2 * vt.data[i] + (1.0f - beta2) * (grad * grad);

            mt.data[i] = mNew;
            vt.data[i] = vNew;

            float mHat = mNew / bc1;
            float vHat = vNew / bc2;

            float update = (lr * mHat) / (fSqrt(vHat) + eps);

            if (weightDecay != 0.0f) update += lr * weightDecay * w.data[i];

            w.data[i] -= update;
        }
    }

    @Override
    public Tensor layerNormBackward(Tensor dXHat, Tensor xHat, Tensor std, int dim) {
        Tensor dX = new Tensor(dXHat.rows, dXHat.cols);

        DeepJExecutor.forRange(0, dXHat.rows, r -> {
            int base = r * dim;
            // std is rows×1: element (r,0) lives at std.data[r]
            float invStd = 1.0f / std.data[r];
            float sumD = 0.0f;
            float sumDXHatXHat = 0.0f;

            for (int c = 0; c < dim; c++) {
                float d = dXHat.data[base + c];
                sumD += d;
                sumDXHatXHat += d * xHat.data[base + c];
            }

            for (int c = 0; c < dim; c++) {
                float d  = dXHat.data[base + c];
                float xh = xHat.data[base + c];
                dX.data[base + c] = invStd * (d - sumD / dim - xh * (sumDXHatXHat / dim));
            }
        });

        return dX;
    }

    // ══════════════════════════════════════════════════════════════════
    //  Data accessors
    // ══════════════════════════════════════════════════════════════════

    public float get(Tensor t, int r, int c)           { return t.data[r * t.cols + c]; }
    public void   set(Tensor t, int r, int c, float v) { t.data[r * t.cols + c] = v; }

    public Tensor getRow(Tensor t, int row) {
        Tensor result = new Tensor(1, t.cols);
        System.arraycopy(t.data, row * t.cols, result.data, 0, t.cols);
        return result;
    }

    public void setRow(Tensor t, int row, Tensor source, int srcRow) {
        System.arraycopy(source.data, srcRow * source.cols, t.data, row * t.cols, t.cols);
    }

    public Tensor sliceRows(Tensor t, int[] rowIndices, int cols) {
        Tensor out = new Tensor(rowIndices.length, cols);
        for (int i = 0; i < rowIndices.length; i++)
            System.arraycopy(t.data, rowIndices[i] * cols, out.data, i * cols, cols);
        return out;
    }

    @Override
    public void scatterAddRows(Tensor target, int[] indices, Tensor grad) {
        for (int i = 0; i < indices.length; i++) {
            int tBase = indices[i] * target.cols;
            int gBase = i * target.cols;
            for (int c = 0; c < target.cols; c++) target.data[tBase + c] += grad.data[gBase + c];
        }
    }

    public Tensor sampleRows(Tensor t, int n, Random rnd) {
        Tensor out = new Tensor(n, t.cols);
        for (int i = 0; i < n; i++) {
            int r = rnd.nextInt(t.rows);
            System.arraycopy(t.data, r * t.cols, out.data, i * t.cols, t.cols);
        }
        return out;
    }

    // ══════════════════════════════════════════════════════════════════
    //  In-place overrides (zero allocation — writes result into input)
    // ══════════════════════════════════════════════════════════════════

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

    // ══════════════════════════════════════════════════════════════════
    //  Debug
    // ══════════════════════════════════════════════════════════════════

    public void print(Tensor t, String label) {
        System.out.println(label);
        for (int r = 0; r < t.rows; r++) {
            int base = r * t.cols;
            for (int c = 0; c < t.cols; c++) System.out.printf("%.4f ", t.data[base + c]);
            System.out.println();
        }
    }
}

