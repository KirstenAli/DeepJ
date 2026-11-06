package io.github.kirstenali.deepj.tensor.cpu;

import io.github.kirstenali.deepj.tensor.CpuRanges;
import io.github.kirstenali.deepj.tensor.MatrixShapeChecks;
import io.github.kirstenali.deepj.tensor.Tensor;

/** Fuses matrix scaling and masking; preserves the existing sentinel and FLOAT32 reduction order. */
final class CpuCausalSoftmax {
    private CpuCausalSoftmax() {}

    static Tensor apply(Tensor input, int width, float scale) {
        MatrixShapeChecks.requireCausal(input, width);
        input.materialize();
        Tensor output = new Tensor(input.matrixRows(), width);
        float[] source = input.rawData(), result = output.rawData();
        CpuRanges.forRange(0, input.matrixRows(), row -> row(source, result, row, width, scale));
        return output;
    }

    private static void row(float[] input, float[] output, int row, int width, float scale) {
        int base = row * width, count = row % width + 1;
        for (int column = 0; column < width; column++) {
            output[base + column] = column < count ? input[base + column] * scale : -1e9f;
        }
        float maximum = CpuReductionOps.rowMax(output, base, width);
        float total = exponentials(output, base, width, maximum);
        CpuReductionOps.rowNormalizeInPlace(output, base, width, total);
    }

    private static float exponentials(float[] output, int base, int width, float maximum) {
        float total = 0;
        for (int column = 0; column < width; column++) {
            float value = CpuElementwiseOps.fExp(output[base + column] - maximum);
            output[base + column] = value;
            total += value;
        }
        return total;
    }
}
