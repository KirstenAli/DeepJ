package io.github.kirstenali.deepj.tensor.cpu;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;

import java.util.Arrays;
import java.util.Random;

import static io.github.kirstenali.deepj.tensor.Tensor.requireSameShape;

abstract class CpuReductionOps extends CpuElementwiseOps {
    @Override
    public Tensor sumRows(Tensor a) {
        Tensor result = new Tensor(1, a.cols);
        for (int r = 0; r < a.rows; r++) {
            int base = r * a.cols;
            for (int c = 0; c < a.cols; c++) {
                result.data[c] += a.data[base + c];
            }
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
            for (int c = 0; c < a.cols; c++) {
                if (a.data[base + c] > max) max = a.data[base + c];
            }
            result.data[r] = max;
        });
        return result;
    }

    @Override
    public float sum(Tensor a) {
        float s = 0.0f;
        for (float v : a.data) {
            s += v;
        }
        return s;
    }

    @Override
    public float sumAbs(Tensor a) {
        float s = 0.0f;
        for (float v : a.data) {
            s += Math.abs(v);
        }
        return s;
    }

    @Override
    public Tensor transpose(Tensor a) {
        Tensor result = new Tensor(a.cols, a.rows);
        DeepJExecutor.forRange(0, a.rows, r -> {
            int aBase = r * a.cols;
            for (int c = 0; c < a.cols; c++) {
                result.data[c * a.rows + r] = a.data[aBase + c];
            }
        });
        return result;
    }

    @Override
    public Tensor clamp(Tensor a, float min, float max) {
        return newUnary(a, x -> Math.min(max, Math.max(min, x)));
    }
    @Override public Tensor sqrt(Tensor a)                          { return newUnary(a, CpuBackend::fSqrt); }
    @Override
    public Tensor pow(Tensor a, float exponent) {
        return newUnary(a, x -> fPow(x, exponent));
    }
    @Override public Tensor neg(Tensor a)                           { return newUnary(a, x -> -x); }
    @Override public Tensor exp(Tensor a)                           { return newUnary(a, CpuBackend::fExp); }
    @Override public Tensor log(Tensor a)                           { return newUnary(a, CpuBackend::fLog); }

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

    static float geluScalar(float x) {
        float c = fSqrt(2.0f / (float) Math.PI);
        float x3 = x * x * x;
        float t = c * (x + 0.044715f * x3);
        return 0.5f * x * (1.0f + fTanh(t));
    }

    static float geluDerivScalar(float x) {
        float c = fSqrt(2.0f / (float) Math.PI);
        float x2 = x * x;
        float x3 = x2 * x;
        float t = c * (x + 0.044715f * x3);
        float tanhT = fTanh(t);
        float sech2 = 1.0f - tanhT * tanhT;
        float dtDx = c * (1.0f + 3.0f * 0.044715f * x2);
        return 0.5f * (1.0f + tanhT) + 0.5f * x * sech2 * dtDx;
    }

    static float rowMax(float[] data, int base, int cols) {
        float max = Float.NEGATIVE_INFINITY;
        for (int c = 0; c < cols; c++) {
            if (data[base + c] > max) max = data[base + c];
        }
        return max;
    }

    static float rowSum(float[] data, int base, int cols) {
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            sum += data[base + c];
        }
        return sum;
    }

    static float rowDot(float[] a, float[] b, int base, int cols) {
        float dot = 0.0f;
        for (int c = 0; c < cols; c++) {
            dot += a[base + c] * b[base + c];
        }
        return dot;
    }

    static float rowSumExpShifted(float[] data, int base, int cols, float max) {
        float sumExp = 0.0f;
        for (int c = 0; c < cols; c++) {
            sumExp += fExp(data[base + c] - max);
        }
        return sumExp;
    }

    static void rowWriteExpShifted(float[] src, float[] out, int base, int cols, float max) {
        for (int c = 0; c < cols; c++) {
            out[base + c] = fExp(src[base + c] - max);
        }
    }

    static void rowNormalizeInPlace(float[] data, int base, int cols, float denom) {
        for (int c = 0; c < cols; c++) {
            data[base + c] /= denom;
        }
    }

    static void rowWriteSoftmax(float[] logits, float[] out, int base, int cols) {
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

            for (int c = 0; c < gradOutput.cols; c++) {
                result.data[base + c] = softmaxOut.data[base + c] * (gradOutput.data[base + c] - dot);
            }
        });

        return result;
    }


}
