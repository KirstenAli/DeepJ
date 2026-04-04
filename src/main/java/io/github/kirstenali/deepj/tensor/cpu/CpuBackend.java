package io.github.kirstenali.deepj.tensor.cpu;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;

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
        DeepJExecutor.forRange(0, rows, r -> {
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
        DeepJExecutor.forRange(0, size, r -> {
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
        DeepJExecutor.forRange(0, flat.length, i -> {
            t.data[i / cols][i % cols] = flat[i];
        });
        return t;
    }

    @Override
    public double[] flattenTensor(Tensor t) {
        double[] flat = new double[t.rows * t.cols];
        DeepJExecutor.forRange(0, t.rows, r -> {
            int base = r * t.cols;
            double[] tr = t.data[r];
            if (t.cols >= 0) System.arraycopy(tr, 0, flat, base, t.cols);
        });
        return flat;
    }

    // ══════════════════════════════════════════════════════════════════
    //  Core binary ops
    // ══════════════════════════════════════════════════════════════════

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        requireMatmulCompatible(a, b);
        Tensor result = new Tensor(a.rows, b.cols);

        DeepJExecutor.forRange(0, a.rows, r -> {
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

        DeepJExecutor.forRange(0, a.rows, r -> {
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

        DeepJExecutor.forRange(0, a.rows, r -> {
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

        DeepJExecutor.forRange(0, a.rows, r -> {
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

        DeepJExecutor.forRange(0, a.rows, r -> {
            double[] ar = a.data[r], br = b.data[r], rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                rr[c] = ar[c] / br[c];
            }
        });

        return result;
    }

    // ══════════════════════════════════════════════════════════════════
    //  Broadcasts
    // ══════════════════════════════════════════════════════════════════

    @Override
    public Tensor addRowVector(Tensor a, Tensor rowVector) {
        if (rowVector.rows != 1 || rowVector.cols != a.cols)
            throw new IllegalArgumentException("rowVector must be 1x" + a.cols);

        Tensor result = new Tensor(a.rows, a.cols);
        double[] rv = rowVector.data[0];

        DeepJExecutor.forRange(0, a.rows, r -> {
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

        DeepJExecutor.forRange(0, a.rows, r -> {
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

        DeepJExecutor.forRange(0, a.rows, r -> {
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

        DeepJExecutor.forRange(0, a.rows, r -> {
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

        DeepJExecutor.forRange(0, a.rows, r -> {
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

        DeepJExecutor.forRange(0, a.rows, r -> {
            double v = colVector.data[r][0];
            double[] ar = a.data[r];
            double[] rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                rr[c] = ar[c] / v;
            }
        });

        return result;
    }

    // ══════════════════════════════════════════════════════════════════
    //  Scalar ops
    // ══════════════════════════════════════════════════════════════════

    @Override
    public Tensor multiplyScalar(Tensor a, double scalar) {
        Tensor result = new Tensor(a.rows, a.cols);

        DeepJExecutor.forRange(0, a.rows, r -> {
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

        DeepJExecutor.forRange(0, a.rows, r -> {
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

        DeepJExecutor.forRange(0, a.rows, r -> {
            double[] ar = a.data[r];
            double[] rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                rr[c] = ar[c] / scalar;
            }
        });

        return result;
    }

    // ══════════════════════════════════════════════════════════════════
    //  Reductions / statistics
    // ══════════════════════════════════════════════════════════════════

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

        DeepJExecutor.forRange(0, a.rows, r -> {
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

        DeepJExecutor.forRange(0, a.rows, r -> {
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

        DeepJExecutor.forRange(0, a.rows, r -> {
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

    @Override
    public Tensor maxAlongRows(Tensor a) {
        Tensor result = new Tensor(a.rows, 1);

        DeepJExecutor.forRange(0, a.rows, r -> {
            double max = Double.NEGATIVE_INFINITY;
            double[] ar = a.data[r];
            for (int c = 0; c < a.cols; c++) {
                if (ar[c] > max) max = ar[c];
            }
            result.data[r][0] = max;
        });

        return result;
    }

    @Override
    public double sum(Tensor a) {
        double s = 0.0;
        for (int r = 0; r < a.rows; r++) {
            double[] row = a.data[r];
            for (int c = 0; c < a.cols; c++) s += row[c];
        }
        return s;
    }

    @Override
    public double sumAbs(Tensor a) {
        double s = 0.0;
        for (int r = 0; r < a.rows; r++) {
            double[] row = a.data[r];
            for (int c = 0; c < a.cols; c++) s += Math.abs(row[c]);
        }
        return s;
    }

    // ══════════════════════════════════════════════════════════════════
    //  Unary math
    // ══════════════════════════════════════════════════════════════════

    @Override
    public Tensor transpose(Tensor a) {
        Tensor result = new Tensor(a.cols, a.rows);

        DeepJExecutor.forRange(0, a.rows, r -> {
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

        DeepJExecutor.forRange(0, a.rows, r -> {
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

        DeepJExecutor.forRange(0, a.rows, r -> {
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

        DeepJExecutor.forRange(0, a.rows, r -> {
            double[] ar = a.data[r];
            double[] rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                rr[c] = Math.pow(ar[c], exponent);
            }
        });

        return result;
    }

    @Override
    public Tensor neg(Tensor a) {
        Tensor result = new Tensor(a.rows, a.cols);

        DeepJExecutor.forRange(0, a.rows, r -> {
            double[] ar = a.data[r];
            double[] rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                rr[c] = -ar[c];
            }
        });

        return result;
    }

    @Override
    public Tensor exp(Tensor a) {
        Tensor result = new Tensor(a.rows, a.cols);

        DeepJExecutor.forRange(0, a.rows, r -> {
            double[] ar = a.data[r];
            double[] rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                rr[c] = Math.exp(ar[c]);
            }
        });

        return result;
    }

    @Override
    public Tensor log(Tensor a) {
        Tensor result = new Tensor(a.rows, a.cols);

        DeepJExecutor.forRange(0, a.rows, r -> {
            double[] ar = a.data[r];
            double[] rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                rr[c] = Math.log(ar[c]);
            }
        });

        return result;
    }

    // ══════════════════════════════════════════════════════════════════
    //  Activation element-wise
    // ══════════════════════════════════════════════════════════════════

    @Override
    public Tensor tanh(Tensor a) {
        Tensor result = new Tensor(a.rows, a.cols);

        DeepJExecutor.forRange(0, a.rows, r -> {
            double[] ar = a.data[r];
            double[] rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                rr[c] = Math.tanh(ar[c]);
            }
        });

        return result;
    }

    @Override
    public Tensor sigmoid(Tensor a) {
        Tensor result = new Tensor(a.rows, a.cols);

        DeepJExecutor.forRange(0, a.rows, r -> {
            double[] ar = a.data[r];
            double[] rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                rr[c] = 1.0 / (1.0 + Math.exp(-ar[c]));
            }
        });

        return result;
    }

    @Override
    public Tensor relu(Tensor a) {
        Tensor result = new Tensor(a.rows, a.cols);

        DeepJExecutor.forRange(0, a.rows, r -> {
            double[] ar = a.data[r];
            double[] rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                rr[c] = Math.max(0.0, ar[c]);
            }
        });

        return result;
    }

    @Override
    public Tensor reluBackward(Tensor input, Tensor gradOutput) {
        requireSameShape(input, gradOutput, "reluBackward");
        Tensor result = new Tensor(input.rows, input.cols);

        DeepJExecutor.forRange(0, input.rows, r -> {
            double[] ir = input.data[r];
            double[] gr = gradOutput.data[r];
            double[] rr = result.data[r];
            for (int c = 0; c < input.cols; c++) {
                rr[c] = ir[c] > 0 ? gr[c] : 0.0;
            }
        });

        return result;
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

    @Override
    public Tensor gelu(Tensor a) {
        Tensor result = new Tensor(a.rows, a.cols);

        DeepJExecutor.forRange(0, a.rows, r -> {
            double[] ar = a.data[r];
            double[] rr = result.data[r];
            for (int c = 0; c < a.cols; c++) {
                rr[c] = geluScalar(ar[c]);
            }
        });

        return result;
    }

    @Override
    public Tensor geluBackward(Tensor input, Tensor gradOutput) {
        requireSameShape(input, gradOutput, "geluBackward");
        Tensor result = new Tensor(input.rows, input.cols);

        DeepJExecutor.forRange(0, input.rows, r -> {
            double[] ir = input.data[r];
            double[] gr = gradOutput.data[r];
            double[] rr = result.data[r];
            for (int c = 0; c < input.cols; c++) {
                rr[c] = gr[c] * geluDerivScalar(ir[c]);
            }
        });

        return result;
    }

    // ══════════════════════════════════════════════════════════════════
    //  Row-wise compound
    // ══════════════════════════════════════════════════════════════════

    @Override
    public Tensor softmaxRows(Tensor logits) {
        Tensor result = new Tensor(logits.rows, logits.cols);

        DeepJExecutor.forRange(0, logits.rows, r -> {
            double[] lr = logits.data[r];
            double[] rr = result.data[r];

            double max = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < logits.cols; c++) {
                if (lr[c] > max) max = lr[c];
            }

            double sumExp = 0.0;
            for (int c = 0; c < logits.cols; c++) {
                double e = Math.exp(lr[c] - max);
                rr[c] = e;
                sumExp += e;
            }

            for (int c = 0; c < logits.cols; c++) {
                rr[c] /= sumExp;
            }
        });

        return result;
    }

    @Override
    public Tensor softmaxBackward(Tensor gradOutput, Tensor softmaxOut) {
        requireSameShape(gradOutput, softmaxOut, "softmaxBackward");
        Tensor result = new Tensor(gradOutput.rows, gradOutput.cols);

        DeepJExecutor.forRange(0, gradOutput.rows, r -> {
            double[] gr = gradOutput.data[r];
            double[] pr = softmaxOut.data[r];
            double[] rr = result.data[r];

            double dot = 0.0;
            for (int c = 0; c < gradOutput.cols; c++) {
                dot += gr[c] * pr[c];
            }

            for (int c = 0; c < gradOutput.cols; c++) {
                rr[c] = pr[c] * (gr[c] - dot);
            }
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
            double[] row = logits.data[i];
            int target = targets[i];

            double max = Double.NEGATIVE_INFINITY;
            for (double v : row) if (v > max) max = v;

            double sumExp = 0.0;
            for (double v : row) sumExp += Math.exp(v - max);

            double logDen = Math.log(sumExp) + max;
            lossSum += logDen - row[target];
        }
        return lossSum / logits.rows;
    }

    @Override
    public Tensor crossEntropyGradient(Tensor logits, int[] targets) {
        Tensor.requireTargetsMatchRows(logits, targets);

        Tensor grad = new Tensor(logits.rows, logits.cols);

        for (int i = 0; i < logits.rows; i++) {
            double[] lr = logits.data[i];
            double[] gr = grad.data[i];

            // softmax
            double max = Double.NEGATIVE_INFINITY;
            for (double v : lr) if (v > max) max = v;
            double sumExp = 0.0;
            for (double v : lr) sumExp += Math.exp(v - max);
            for (int c = 0; c < logits.cols; c++) {
                gr[c] = Math.exp(lr[c] - max) / sumExp;
            }
            gr[targets[i]] -= 1.0;
        }

        return divideScalar(grad, logits.rows);
    }

    @Override
    public void adamWUpdate(Tensor w, Tensor g, Tensor mt, Tensor vt,
                            double lr, double beta1, double beta2, double eps,
                            double weightDecay, double bc1, double bc2) {
        for (int r = 0; r < w.rows; r++) {
            for (int c = 0; c < w.cols; c++) {
                double grad = g.data[r][c];

                double mNew = beta1 * mt.data[r][c] + (1.0 - beta1) * grad;
                double vNew = beta2 * vt.data[r][c] + (1.0 - beta2) * (grad * grad);

                mt.data[r][c] = mNew;
                vt.data[r][c] = vNew;

                double mHat = mNew / bc1;
                double vHat = vNew / bc2;

                double update = (lr * mHat) / (Math.sqrt(vHat) + eps);

                if (weightDecay != 0.0) {
                    update += lr * weightDecay * w.data[r][c];
                }

                w.data[r][c] -= update;
            }
        }
    }

    @Override
    public Tensor layerNormBackward(Tensor dXHat, Tensor xHat, Tensor std, int dim) {
        Tensor dX = new Tensor(dXHat.rows, dXHat.cols);

        DeepJExecutor.forRange(0, dXHat.rows, r -> {
            double stdR = std.data[r][0];
            double invStd = 1.0 / stdR;
            double sumD = 0.0;
            double sumDXHatXHat = 0.0;

            for (int c = 0; c < dim; c++) {
                double d = dXHat.data[r][c];
                sumD += d;
                sumDXHatXHat += d * xHat.data[r][c];
            }

            for (int c = 0; c < dim; c++) {
                double d = dXHat.data[r][c];
                double xh = xHat.data[r][c];
                dX.data[r][c] = invStd * (d - sumD / dim - xh * (sumDXHatXHat / dim));
            }
        });

        return dX;
    }

    // ══════════════════════════════════════════════════════════════════
    //  Data accessors
    // ══════════════════════════════════════════════════════════════════

    @Override
    public double get(Tensor t, int r, int c) {
        return t.data[r][c];
    }

    @Override
    public void set(Tensor t, int r, int c, double value) {
        t.data[r][c] = value;
    }

    @Override
    public Tensor getRow(Tensor t, int row) {
        Tensor result = new Tensor(1, t.cols);
        System.arraycopy(t.data[row], 0, result.data[0], 0, t.cols);
        return result;
    }

    @Override
    public void setRow(Tensor t, int row, Tensor source, int srcRow) {
        System.arraycopy(source.data[srcRow], 0, t.data[row], 0, t.cols);
    }

    @Override
    public Tensor sliceRows(Tensor t, int[] rowIndices, int cols) {
        Tensor out = new Tensor(rowIndices.length, cols);
        for (int i = 0; i < rowIndices.length; i++) {
            System.arraycopy(t.data[rowIndices[i]], 0, out.data[i], 0, cols);
        }
        return out;
    }

    @Override
    public void scatterAddRows(Tensor target, int[] indices, Tensor grad) {
        for (int i = 0; i < indices.length; i++) {
            int id = indices[i];
            double[] tRow = target.data[id];
            double[] gRow = grad.data[i];
            for (int c = 0; c < target.cols; c++) {
                tRow[c] += gRow[c];
            }
        }
    }

    @Override
    public Tensor sampleRows(Tensor t, int n, Random rnd) {
        Tensor out = new Tensor(n, t.cols);
        for (int i = 0; i < n; i++) {
            int r = rnd.nextInt(t.rows);
            System.arraycopy(t.data[r], 0, out.data[i], 0, t.cols);
        }
        return out;
    }

    // ══════════════════════════════════════════════════════════════════
    //  Debug
    // ══════════════════════════════════════════════════════════════════

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