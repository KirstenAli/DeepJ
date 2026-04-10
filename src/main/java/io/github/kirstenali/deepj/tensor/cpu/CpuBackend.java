package io.github.kirstenali.deepj.tensor.cpu;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;

import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

import static io.github.kirstenali.deepj.tensor.Tensor.requireSameShape;

public final class CpuBackend implements TensorBackend {

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

    static void applyUnary(Tensor a, Tensor out, DoubleUnaryOperator fn) {
        DeepJExecutor.forRange(0, a.rows, r -> {
            int base = r * a.cols;
            for (int c = 0; c < a.cols; c++) out.data[base + c] = (float) fn.applyAsDouble(a.data[base + c]);
        });
    }

    private static Tensor newUnary(Tensor a, DoubleUnaryOperator fn) {
        Tensor out = new Tensor(a.rows, a.cols);
        applyUnary(a, out, fn);
        return out;
    }

    // ── binary: out[i] = fn(a[i], b[i]) ───────────────────────────

    static void applyBinary(Tensor a, Tensor b, Tensor out, DoubleBinaryOperator fn) {
        DeepJExecutor.forRange(0, a.rows, r -> {
            int base = r * a.cols;
            for (int c = 0; c < a.cols; c++) out.data[base + c] = (float) fn.applyAsDouble(a.data[base + c], b.data[base + c]);
        });
    }

    private static Tensor newBinary(Tensor a, Tensor b, DoubleBinaryOperator fn) {
        Tensor out = new Tensor(a.rows, a.cols);
        applyBinary(a, b, out, fn);
        return out;
    }

    // ── scalar: out[i] = fn(a[i], scalar) ────────────────────────

    static void applyScalar(Tensor a, double s, Tensor out, DoubleBinaryOperator fn) {
        DeepJExecutor.forRange(0, a.rows, r -> {
            int base = r * a.cols;
            for (int c = 0; c < a.cols; c++) out.data[base + c] = (float) fn.applyAsDouble(a.data[base + c], s);
        });
    }

    private static Tensor newScalar(Tensor a, double s, DoubleBinaryOperator fn) {
        Tensor out = new Tensor(a.rows, a.cols);
        applyScalar(a, s, out, fn);
        return out;
    }

    // ── column broadcast: out[r,c] = fn(a[r,c], col[r,0]) ─────────
    //    col is rows×1, so col.data[r] is element (r,0)

    static void applyColBroadcast(Tensor a, Tensor col, Tensor out, DoubleBinaryOperator fn) {
        DeepJExecutor.forRange(0, a.rows, r -> {
            double v = col.data[r]; // col is Rx1 → data[r*1+0] = data[r]
            int base = r * a.cols;
            for (int c = 0; c < a.cols; c++) out.data[base + c] = (float) fn.applyAsDouble(a.data[base + c], v);
        });
    }

    private static Tensor newColBroadcast(Tensor a, Tensor col, DoubleBinaryOperator fn) {
        Tensor out = new Tensor(a.rows, a.cols);
        applyColBroadcast(a, col, out, fn);
        return out;
    }

    // ── row broadcast: out[r,c] = fn(a[r,c], row[0,c]) ────────────
    //    row is 1×cols, so row.data[c] is element (0,c)

    static void applyRowBroadcast(Tensor a, Tensor row, Tensor out, DoubleBinaryOperator fn) {
        DeepJExecutor.forRange(0, a.rows, r -> {
            int base = r * a.cols;
            for (int c = 0; c < a.cols; c++) out.data[base + c] = (float) fn.applyAsDouble(a.data[base + c], row.data[c]);
        });
    }

    private static Tensor newRowBroadcast(Tensor a, Tensor row, DoubleBinaryOperator fn) {
        Tensor out = new Tensor(a.rows, a.cols);
        applyRowBroadcast(a, row, out, fn);
        return out;
    }

    // ══════════════════════════════════════════════════════════════════
    //  Factories
    // ══════════════════════════════════════════════════════════════════

    @Override
    public Tensor zeros(int rows, int cols) {
        return new Tensor(rows, cols);
    }

    @Override
    public Tensor ones(int rows, int cols) {
        Tensor result = new Tensor(rows, cols);
        Arrays.fill(result.data, 1.0f);
        return result;
    }

    @Override
    public Tensor random(int rows, int cols, Random rand) {
        Tensor t = new Tensor(rows, cols);
        for (int i = 0; i < t.data.length; i++) t.data[i] = (float) (rand.nextGaussian() * 0.1);
        return t;
    }

    @Override
    public Tensor causalMask(int size) {
        Tensor mask = new Tensor(size, size);
        DeepJExecutor.forRange(0, size, r -> {
            int base = r * size;
            for (int c = 0; c < size; c++) mask.data[base + c] = (c > r) ? -1e9f : 0.0f;
        });
        return mask;
    }

    @Override
    public Tensor unflattenToTensor(float[] flat, int rows, int cols) {
        Tensor t = new Tensor(rows, cols);
        System.arraycopy(flat, 0, t.data, 0, flat.length);
        return t;
    }

    @Override
    public float[] flattenTensor(Tensor t) {
        return Arrays.copyOf(t.data, t.data.length);
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
                double aVal = a.data[aBase + k];
                int bBase = k * b.cols;
                for (int c = 0; c < b.cols; c++) {
                    result.data[outBase + c] += aVal * b.data[bBase + c];
                }
            }
        });

        return result;
    }

    @Override public Tensor add(Tensor a, Tensor b)      { requireSameShape(a, b, "add");      return newBinary(a, b, Double::sum); }
    @Override public Tensor subtract(Tensor a, Tensor b)  { requireSameShape(a, b, "subtract"); return newBinary(a, b, (x, y) -> x - y); }
    @Override public Tensor multiply(Tensor a, Tensor b)  { requireSameShape(a, b, "multiply"); return newBinary(a, b, (x, y) -> x * y); }
    @Override public Tensor divide(Tensor a, Tensor b)    { requireSameShape(a, b, "divide");   return newBinary(a, b, (x, y) -> x / y); }

    // ══════════════════════════════════════════════════════════════════
    //  Broadcasts
    // ══════════════════════════════════════════════════════════════════

    @Override public Tensor addRowVector(Tensor a, Tensor rv)          { requireRowVector(rv, a); return newRowBroadcast(a, rv, Double::sum); }
    @Override public Tensor addBroadcastRows(Tensor a, Tensor rv)      { return addRowVector(a, rv); }
    @Override public Tensor multiplyBroadcastRows(Tensor a, Tensor rv) { requireRowVector(rv, a); return newRowBroadcast(a, rv, (x, y) -> x * y); }

    @Override public Tensor addBroadcastCols(Tensor a, Tensor cv)      { requireColVector(cv, a); return newColBroadcast(a, cv, Double::sum); }
    @Override public Tensor subtractBroadcastCols(Tensor a, Tensor cv)  { requireColVector(cv, a); return newColBroadcast(a, cv, (x, y) -> x - y); }
    @Override public Tensor multiplyBroadcastCols(Tensor a, Tensor cv)  { requireColVector(cv, a); return newColBroadcast(a, cv, (x, y) -> x * y); }
    @Override public Tensor divideBroadcastCols(Tensor a, Tensor cv)    { requireColVector(cv, a); return newColBroadcast(a, cv, (x, y) -> x / y); }

    // ══════════════════════════════════════════════════════════════════
    //  Scalar ops
    // ══════════════════════════════════════════════════════════════════

    @Override public Tensor multiplyScalar(Tensor a, double s) { return newScalar(a, s, (x, v) -> x * v); }
    @Override public Tensor addScalar(Tensor a, double s)      { return newScalar(a, s, Double::sum); }
    @Override public Tensor divideScalar(Tensor a, double s)   { return newScalar(a, s, (x, v) -> x / v); }

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
            double sum = 0.0;
            int base = r * a.cols;
            for (int c = 0; c < a.cols; c++) sum += a.data[base + c];
            result.data[r] = (float) sum;
        });
        return result;
    }

    @Override public Tensor sumAlongCols(Tensor a) { return sumRows(a); }

    @Override
    public Tensor meanAlongRows(Tensor a) {
        Tensor result = new Tensor(a.rows, 1);
        double invCols = 1.0 / a.cols;
        DeepJExecutor.forRange(0, a.rows, r -> {
            double sum = 0.0;
            int base = r * a.cols;
            for (int c = 0; c < a.cols; c++) sum += a.data[base + c];
            result.data[r] = (float) (sum * invCols);
        });
        return result;
    }

    @Override
    public Tensor varianceAlongRows(Tensor a) {
        Tensor result = new Tensor(a.rows, 1);
        double invCols = 1.0 / a.cols;
        DeepJExecutor.forRange(0, a.rows, r -> {
            int base = r * a.cols;
            double sum = 0.0;
            for (int c = 0; c < a.cols; c++) sum += a.data[base + c];
            double mean = sum * invCols;
            double acc = 0.0;
            for (int c = 0; c < a.cols; c++) {
                double diff = a.data[base + c] - mean;
                acc += diff * diff;
            }
            result.data[r] = (float) (acc * invCols);
        });
        return result;
    }

    @Override
    public Tensor maxAlongRows(Tensor a) {
        Tensor result = new Tensor(a.rows, 1);
        DeepJExecutor.forRange(0, a.rows, r -> {
            int base = r * a.cols;
            double max = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < a.cols; c++) if (a.data[base + c] > max) max = a.data[base + c];
            result.data[r] = (float) max;
        });
        return result;
    }

    @Override
    public double sum(Tensor a) {
        double s = 0.0;
        for (float v : a.data) s += v;
        return s;
    }

    @Override
    public double sumAbs(Tensor a) {
        double s = 0.0;
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

    @Override public Tensor clamp(Tensor a, double min, double max) { return newUnary(a, x -> Math.min(max, Math.max(min, x))); }
    @Override public Tensor sqrt(Tensor a)                          { return newUnary(a, Math::sqrt); }
    @Override public Tensor pow(Tensor a, double exponent)          { return newUnary(a, x -> Math.pow(x, exponent)); }
    @Override public Tensor neg(Tensor a)                           { return newUnary(a, x -> -x); }
    @Override public Tensor exp(Tensor a)                           { return newUnary(a, Math::exp); }
    @Override public Tensor log(Tensor a)                           { return newUnary(a, Math::log); }

    // ══════════════════════════════════════════════════════════════════
    //  Activations
    // ══════════════════════════════════════════════════════════════════

    @Override public Tensor tanh(Tensor a)    { return newUnary(a, Math::tanh); }
    @Override public Tensor sigmoid(Tensor a) { return newUnary(a, x -> 1.0 / (1.0 + Math.exp(-x))); }
    @Override public Tensor relu(Tensor a)    { return newUnary(a, x -> Math.max(0.0, x)); }
    @Override public Tensor gelu(Tensor a)    { return newUnary(a, CpuBackend::geluScalar); }

    @Override
    public Tensor reluBackward(Tensor input, Tensor gradOutput) {
        requireSameShape(input, gradOutput, "reluBackward");
        return newBinary(input, gradOutput, (x, g) -> x > 0 ? g : 0.0);
    }

    @Override
    public Tensor geluBackward(Tensor input, Tensor gradOutput) {
        requireSameShape(input, gradOutput, "geluBackward");
        return newBinary(input, gradOutput, (x, g) -> g * geluDerivScalar(x));
    }

    private static double geluScalar(double x) {
        double c = Math.sqrt(2.0 / Math.PI);
        double x3 = x * x * x;
        double t = c * (x + 0.044715 * x3);
        return 0.5 * x * (1.0 + Math.tanh(t));
    }

    private static double geluDerivScalar(double x) {
        double c = Math.sqrt(2.0 / Math.PI);
        double x2 = x * x;
        double x3 = x2 * x;
        double t = c * (x + 0.044715 * x3);
        double tanhT = Math.tanh(t);
        double sech2 = 1.0 - tanhT * tanhT;
        double dt_dx = c * (1.0 + 3.0 * 0.044715 * x2);
        return 0.5 * (1.0 + tanhT) + 0.5 * x * sech2 * dt_dx;
    }

    // ══════════════════════════════════════════════════════════════════
    //  Row-wise compound
    // ══════════════════════════════════════════════════════════════════

    @Override
    public Tensor softmaxRows(Tensor logits) {
        Tensor result = new Tensor(logits.rows, logits.cols);

        DeepJExecutor.forRange(0, logits.rows, r -> {
            int base = r * logits.cols;

            double max = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < logits.cols; c++) if (logits.data[base + c] > max) max = logits.data[base + c];

            double sumExp = 0.0;
            for (int c = 0; c < logits.cols; c++) {
                double e = Math.exp(logits.data[base + c] - max);
                result.data[base + c] = (float) e;
                sumExp += e;
            }

            for (int c = 0; c < logits.cols; c++) result.data[base + c] /= sumExp;
        });

        return result;
    }

    @Override
    public Tensor softmaxBackward(Tensor gradOutput, Tensor softmaxOut) {
        requireSameShape(gradOutput, softmaxOut, "softmaxBackward");
        Tensor result = new Tensor(gradOutput.rows, gradOutput.cols);

        DeepJExecutor.forRange(0, gradOutput.rows, r -> {
            int base = r * gradOutput.cols;

            double dot = 0.0;
            for (int c = 0; c < gradOutput.cols; c++) dot += gradOutput.data[base + c] * softmaxOut.data[base + c];

            for (int c = 0; c < gradOutput.cols; c++)
                result.data[base + c] = (float) (softmaxOut.data[base + c] * (gradOutput.data[base + c] - dot));
        });

        return result;
    }

    // ══════════════════════════════════════════════════════════════════
    //  Fused high-level ops
    // ══════════════════════════════════════════════════════════════════

    @Override
    public double crossEntropyLoss(Tensor logits, int[] targets) {
        Tensor.requireTargetsMatchRows(logits, targets);

        double lossSum = 0.0;
        for (int i = 0; i < logits.rows; i++) {
            int base = i * logits.cols;
            int target = targets[i];

            double max = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < logits.cols; c++) if (logits.data[base + c] > max) max = logits.data[base + c];

            double sumExp = 0.0;
            for (int c = 0; c < logits.cols; c++) sumExp += Math.exp(logits.data[base + c] - max);

            lossSum += Math.log(sumExp) + max - logits.data[base + target];
        }
        return lossSum / logits.rows;
    }

    @Override
    public Tensor crossEntropyGradient(Tensor logits, int[] targets) {
        Tensor.requireTargetsMatchRows(logits, targets);

        Tensor grad = new Tensor(logits.rows, logits.cols);

        for (int i = 0; i < logits.rows; i++) {
            int base = i * logits.cols;

            double max = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < logits.cols; c++) if (logits.data[base + c] > max) max = logits.data[base + c];
            double sumExp = 0.0;
            for (int c = 0; c < logits.cols; c++) sumExp += Math.exp(logits.data[base + c] - max);
            for (int c = 0; c < logits.cols; c++) grad.data[base + c] = (float) (Math.exp(logits.data[base + c] - max) / sumExp);
            grad.data[base + targets[i]] -= 1.0f;
        }

        divideScalarInPlace(grad, logits.rows);
        return grad;
    }

    @Override
    public void adamWUpdate(Tensor w, Tensor g, Tensor mt, Tensor vt,
                            double lr, double beta1, double beta2, double eps,
                            double weightDecay, double bc1, double bc2) {
        int n = w.data.length;
        for (int i = 0; i < n; i++) {
            double grad = g.data[i];

            double mNew = beta1 * mt.data[i] + (1.0 - beta1) * grad;
            double vNew = beta2 * vt.data[i] + (1.0 - beta2) * (grad * grad);

            mt.data[i] = (float) mNew;
            vt.data[i] = (float) vNew;

            double mHat = mNew / bc1;
            double vHat = vNew / bc2;

            double update = (lr * mHat) / (Math.sqrt(vHat) + eps);

            if (weightDecay != 0.0) update += lr * weightDecay * w.data[i];

            w.data[i] -= (float) update;
        }
    }

    @Override
    public Tensor layerNormBackward(Tensor dXHat, Tensor xHat, Tensor std, int dim) {
        Tensor dX = new Tensor(dXHat.rows, dXHat.cols);

        DeepJExecutor.forRange(0, dXHat.rows, r -> {
            int base = r * dim;
            // std is rows×1: element (r,0) lives at std.data[r]
            double invStd = 1.0 / std.data[r];
            double sumD = 0.0;
            double sumDXHatXHat = 0.0;

            for (int c = 0; c < dim; c++) {
                double d = dXHat.data[base + c];
                sumD += d;
                sumDXHatXHat += d * xHat.data[base + c];
            }

            for (int c = 0; c < dim; c++) {
                double d  = dXHat.data[base + c];
                double xh = xHat.data[base + c];
                dX.data[base + c] = (float) (invStd * (d - sumD / dim - xh * (sumDXHatXHat / dim)));
            }
        });

        return dX;
    }

    // ══════════════════════════════════════════════════════════════════
    //  Data accessors
    // ══════════════════════════════════════════════════════════════════

    @Override public double get(Tensor t, int r, int c)           { return t.data[r * t.cols + c]; }
    @Override public void   set(Tensor t, int r, int c, double v) { t.data[r * t.cols + c] = (float) v; }

    @Override
    public Tensor getRow(Tensor t, int row) {
        Tensor result = new Tensor(1, t.cols);
        System.arraycopy(t.data, row * t.cols, result.data, 0, t.cols);
        return result;
    }

    @Override
    public void setRow(Tensor t, int row, Tensor source, int srcRow) {
        System.arraycopy(source.data, srcRow * source.cols, t.data, row * t.cols, t.cols);
    }

    @Override
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

    @Override
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

    @Override public void addInPlace(Tensor a, Tensor b)           { applyBinary(a, b, a, Double::sum); }
    @Override public void subtractInPlace(Tensor a, Tensor b)      { applyBinary(a, b, a, (x, y) -> x - y); }
    @Override public void multiplyInPlace(Tensor a, Tensor b)      { applyBinary(a, b, a, (x, y) -> x * y); }
    @Override public void divideInPlace(Tensor a, Tensor b)        { applyBinary(a, b, a, (x, y) -> x / y); }

    @Override public void multiplyScalarInPlace(Tensor a, double s) { applyScalar(a, s, a, (x, v) -> x * v); }
    @Override public void addScalarInPlace(Tensor a, double s)      { applyScalar(a, s, a, Double::sum); }
    @Override public void divideScalarInPlace(Tensor a, double s)   { applyScalar(a, s, a, (x, v) -> x / v); }

    @Override public void sqrtInPlace(Tensor a)    { applyUnary(a, a, Math::sqrt); }
    @Override public void negInPlace(Tensor a)     { applyUnary(a, a, x -> -x); }
    @Override public void expInPlace(Tensor a)     { applyUnary(a, a, Math::exp); }
    @Override public void logInPlace(Tensor a)     { applyUnary(a, a, Math::log); }
    @Override public void reluInPlace(Tensor a)    { applyUnary(a, a, x -> Math.max(0.0, x)); }
    @Override public void geluInPlace(Tensor a)    { applyUnary(a, a, CpuBackend::geluScalar); }
    @Override public void tanhInPlace(Tensor a)    { applyUnary(a, a, Math::tanh); }
    @Override public void sigmoidInPlace(Tensor a) { applyUnary(a, a, x -> 1.0 / (1.0 + Math.exp(-x))); }

    // ══════════════════════════════════════════════════════════════════
    //  Debug
    // ══════════════════════════════════════════════════════════════════

    @Override
    public void print(Tensor t, String label) {
        System.out.println(label);
        for (int r = 0; r < t.rows; r++) {
            int base = r * t.cols;
            for (int c = 0; c < t.cols; c++) System.out.printf("%.4f ", t.data[base + c]);
            System.out.println();
        }
    }
}

