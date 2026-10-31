package io.github.kirstenali.deepj.tensor;

import java.util.Random;

/** Checked CPU row kernels; the Tensor facade owns autograd mutation fences as for other backends. */
public final class CpuRowAccess {
    private CpuRowAccess() {}

    public static Tensor select(Tensor input, int[] rows, int columns) {
        MatrixShapeChecks.requireMatrix(input);
        if (columns < 0 || columns > input.matrixCols()) {
            throw new IllegalArgumentException("Selected row width exceeds the input");
        }
        Tensor selected = CpuNdIndexing.select(input, 0, rows);
        return columns == input.matrixCols() ? selected
                : new Tensor(TensorViews.slice(selected, 1, 0, columns, 1));
    }

    public static void assign(Tensor target, int row, Tensor source, int sourceRow) {
        target.layout().requireWritable();
        requireRow(target, row);
        requireRow(source, sourceRow);
        if (target.matrixCols() != source.matrixCols()) {
            throw new IllegalArgumentException("Assigned rows must have the same width");
        }
        Tensor destination = TensorViews.slice(target, 0, row, row + 1, 1);
        CpuNdAssignment.apply(destination, TensorViews.slice(source, 0, sourceRow, sourceRow + 1, 1));
        TensorStorage.markGpuNeedsUpload(target);
    }

    public static Tensor sample(Tensor input, int count, Random random) {
        MatrixShapeChecks.requireMatrix(input);
        if (count < 0) throw new IllegalArgumentException("Sample count cannot be negative");
        int[] rows = new int[count];
        for (int index = 0; index < count; index++) { rows[index] = random.nextInt(input.matrixRows()); }
        return CpuNdIndexing.select(input, 0, rows);
    }

    private static void requireRow(Tensor input, int row) {
        if (row < 0 || row >= input.matrixRows()) throw new IndexOutOfBoundsException("Row index: " + row);
    }
}
