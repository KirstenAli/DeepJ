package io.github.kirstenali.deepj.tensor.cpu;

import io.github.kirstenali.deepj.tensor.CrossEntropyChecks;
import io.github.kirstenali.deepj.tensor.CrossEntropyResult;
import io.github.kirstenali.deepj.tensor.Tensor;

/** Dense CPU loss and gradient share exponentials while retaining scalar reduction order. */
final class CpuMatrixCrossEntropy {
    private CpuMatrixCrossEntropy() {}

    static CrossEntropyResult apply(Tensor logits, int[] targets, boolean[] mask) {
        CrossEntropyChecks.requireTargets(logits, targets);
        int rows = logits.matrixRows(), columns = logits.matrixCols();
        int included = mask == null ? rows : CrossEntropyChecks.included(mask, rows);
        CpuElementwiseOps.prepare(logits);
        Tensor gradient = new Tensor(rows, columns);
        float sum = 0;
        for (int row = 0; row < rows; row++) {
            if (mask == null || mask[row]) {
                sum += row(logits.rawData(), gradient.rawData(), row * columns, columns, targets[row], included);
            }
        }
        return new CrossEntropyResult(Tensor.from(new float[]{sum / included}, 1, 1), gradient);
    }

    private static float row(float[] logits, float[] gradient, int base, int columns, int target, int included) {
        float maximum = CpuReductionOps.rowMax(logits, base, columns);
        CpuReductionOps.rowWriteExpShifted(logits, gradient, base, columns, maximum);
        float denominator = CpuReductionOps.rowSum(gradient, base, columns);
        CpuReductionOps.rowNormalizeInPlace(gradient, base, columns, denominator);
        gradient[base + target] -= 1.0f;
        for (int column = 0; column < columns; column++) {
            gradient[base + column] /= included;
        }
        return CpuElementwiseOps.fLog(denominator) + maximum - logits[base + target];
    }
}
