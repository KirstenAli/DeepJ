package io.github.kirstenali.deepj.testing;

import io.github.kirstenali.deepj.tensor.CausalAttentionBackend;
import io.github.kirstenali.deepj.tensor.CausalAttentionResult;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;

/** Independent full-sequence local-causal attention, with no KV cache or circular storage. */
public final class SlidingWindowReference {
    private SlidingWindowReference() {}

    public static CausalAttentionBackend backend(int window) {
        var cpu = new CpuBackend();
        return (CausalAttentionBackend) Proxy.newProxyInstance(CausalAttentionBackend.class.getClassLoader(),
                new Class<?>[]{CausalAttentionBackend.class}, (proxy, method, args) -> {
                    if (method.getName().equals("supportsCausalAttention")) return true;
                    if (method.getName().equals("causalAttention")) return attention(args, window);
                    try { return method.invoke(cpu, args); }
                    catch (InvocationTargetException failure) { throw failure.getCause(); }
                });
    }

    private static CausalAttentionResult attention(Object[] args, int window) {
        Tensor queries = (Tensor) args[0], keys = (Tensor) args[1], values = (Tensor) args[2];
        int heads = (int) args[3], keyHeads = args.length == 6 ? (int) args[4] : heads;
        float scale = (float) args[args.length - 1];
        Tensor output = Tensor.zeros(queries.rows, values.cols);
        for (int row = 0; row < queries.rows; row++) {
            attendRow(queries, keys, values, output, row, heads / keyHeads, queries.rows / heads, window, scale);
        }
        return new CausalAttentionResult(output, Tensor.zeros(queries.rows, 2));
    }

    private static void attendRow(Tensor query, Tensor key, Tensor value, Tensor output,
                                   int row, int groups, int length, int window, float scale) {
        int position = row % length, first = Math.max(0, position - window + 1);
        int keyStart = row / length / groups * length;
        double[] scores = new double[position - first + 1];
        for (int index = 0; index < scores.length; index++) {
            scores[index] = dot(query, row, key, keyStart + first + index) * scale;
        }
        double total = normalize(scores);
        for (int column = 0; column < output.cols; column++) {
            output.rawData()[row * output.cols + column] = weighted(value, keyStart + first, column, scores, total);
        }
    }

    private static double dot(Tensor left, int row, Tensor right, int otherRow) {
        double sum = 0;
        for (int column = 0; column < left.cols; column++) {
            sum += (double) left.rawData()[row * left.cols + column] * right.rawData()[otherRow * right.cols + column];
        }
        return sum;
    }

    private static double normalize(double[] scores) {
        double max = java.util.Arrays.stream(scores).max().orElseThrow(), total = 0;
        for (int index = 0; index < scores.length; index++) {
            scores[index] = Math.exp(scores[index] - max);
            total += scores[index];
        }
        return total;
    }

    private static float weighted(Tensor values, int first, int column, double[] scores, double total) {
        double sum = 0;
        for (int index = 0; index < scores.length; index++) {
            sum += scores[index] * values.rawData()[(first + index) * values.cols + column];
        }
        return (float) (sum / total);
    }
}
