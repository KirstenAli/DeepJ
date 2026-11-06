package io.github.kirstenali.deepj.testing;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.Arrays;
import java.util.Map;

/** Optional CPU matrix probe; separate JVMs can use the same harness against old/new core JARs. */
public final class CpuMatmulBenchmark {
    private CpuMatmulBenchmark() {}

    public static void main(String[] args) {
        Tensor.setBackend(new CpuBackend());
        DeepJExecutor.setNumThreads(Integer.parseInt(args[1]));
        DType dtype = DType.valueOf(args[0]);
        int rows = Integer.parseInt(args[2]), inner = Integer.parseInt(args[3]), columns = Integer.parseInt(args[4]);
        boolean transpose = Boolean.parseBoolean(args[5]);
        Tensor a = input(rows, inner, dtype);
        Tensor b = input(transpose ? columns : inner, transpose ? inner : columns, dtype);
        try { measure(a, b, transpose, args); }
        finally { DeepJExecutor.shutdown(); }
    }

    private static void measure(Tensor left, Tensor right, boolean transpose, String[] args) {
        double[] milliseconds = new double[7];
        Tensor output = null;
        for (int iteration = -5; iteration < milliseconds.length; iteration++) {
            long start = System.nanoTime();
            output = transpose ? Tensor.backend().batchedMatmul(left, right, 1, false, true) : left.matmul(right);
            if (iteration >= 0) milliseconds[iteration] = (System.nanoTime() - start) / 1e6;
        }
        double checksum = 0;
        for (float value : output.toFloatArray()) { checksum += value; }
        Arrays.sort(milliseconds);
        System.out.println(Qwen3Json.stringify(Map.of("dtype", args[0], "threads", args[1], "rows", args[2],
                "inner", args[3], "columns", args[4], "transpose_right", transpose,
                "median_ms", milliseconds[3], "min_ms", milliseconds[0], "checksum", checksum)));
    }

    private static Tensor input(int rows, int columns, DType dtype) {
        float[] values = new float[rows * columns];
        for (int i = 0; i < values.length; i++) { values[i] = ((i % 31 * 6) % 31 - 15) * .015625f; }
        return Tensor.from(values, dtype, rows, columns);
    }
}
