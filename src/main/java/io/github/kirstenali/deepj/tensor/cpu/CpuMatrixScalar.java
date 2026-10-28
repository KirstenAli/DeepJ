package io.github.kirstenali.deepj.tensor.cpu;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import io.github.kirstenali.deepj.tensor.Tensor;

/** Direct matrix scaling keeps the per-value arithmetic independent of callback profiles. */
final class CpuMatrixScalar {
    private CpuMatrixScalar() {}

    static void multiply(Tensor input, float scalar) {
        CpuElementwiseOps.prepare(input);
        float[] values = input.rawData();
        int columns = input.matrixCols();
        DeepJExecutor.forRange(0, input.matrixRows(), row -> multiplyRow(values, row * columns, columns, scalar));
    }

    private static void multiplyRow(float[] values, int first, int columns, float scalar) {
        for (int column = 0; column < columns; column++) {
            values[first + column] *= scalar;
        }
    }
}
