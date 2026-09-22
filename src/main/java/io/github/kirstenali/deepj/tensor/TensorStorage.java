package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;

import java.util.Arrays;
import java.util.Random;

final class TensorStorage {

    private static final CpuBackend CPU = new CpuBackend();

    private TensorStorage() {}

    static int checkedSize(int rows, int cols) {
        if (rows < 1 || cols < 1) throw new IllegalArgumentException("Tensor dimensions must be positive");
        try {
            return Math.multiplyExact(rows, cols);
        } catch (ArithmeticException error) {
            throw new IllegalArgumentException("Tensor shape is too large: " + rows + "x" + cols, error);
        }
    }

    static float[] copyData(Tensor source) {
        source.materialize();
        return Arrays.copyOf(source.data, source.data.length);
    }

    static void requireSource(Tensor source) {
        if (source == null) throw new IllegalArgumentException("source cannot be null");
    }

    static float[] rowData(Tensor tensor, int row) {
        requireRow(tensor, row);
        tensor.materialize();
        return Arrays.copyOfRange(tensor.data, row * tensor.cols, (row + 1) * tensor.cols);
    }

    static void materialize(Tensor tensor) {
        if (tensor.gpuTag != null) Tensor.backend().materializeTensor(tensor);
    }

    static Tensor from2D(float[][] data) {
        requireMatrix(data);
        Tensor tensor = new Tensor(data.length, data[0].length);
        copyRows(data, tensor);
        return tensor;
    }

    private static void requireMatrix(float[][] data) {
        if (data == null || data.length == 0 || data[0] == null || data[0].length == 0) {
            throw new IllegalArgumentException("Tensor data must contain at least one value");
        }
    }

    private static void copyRows(float[][] data, Tensor tensor) {
        for (int row = 0; row < data.length; row++) {
            requireRowWidth(data[row], tensor.cols);
            System.arraycopy(data[row], 0, tensor.data, row * tensor.cols, tensor.cols);
        }
    }

    private static void requireRowWidth(float[] row, int columns) {
        if (row == null || row.length != columns) {
            throw new IllegalArgumentException("All rows must have the same length (expected " + columns + ")");
        }
    }

    static float get(Tensor tensor, int row, int column) {
        requireIndex(tensor, row, column);
        tensor.materialize();
        return CPU.get(tensor, row, column);
    }

    static void set(Tensor tensor, int row, int column, float value) {
        requireIndex(tensor, row, column);
        tensor.materialize();
        CPU.set(tensor, row, column, value);
        markGpuNeedsUpload(tensor);
    }

    static Tensor getRow(Tensor tensor, int row) {
        requireRow(tensor, row);
        tensor.materialize();
        return CPU.getRow(tensor, row);
    }

    static void setRow(Tensor tensor, int row, Tensor source, int sourceRow) {
        requireRow(tensor, row);
        requireRow(source, sourceRow);
        if (source.cols != tensor.cols) throw new IllegalArgumentException("Source row width must match tensor width");
        tensor.materialize();
        source.materialize();
        CPU.setRow(tensor, row, source, sourceRow);
        markGpuNeedsUpload(tensor);
    }

    static Tensor sliceRows(Tensor tensor, int[] rows, int columns) {
        if (columns != tensor.cols) throw new IllegalArgumentException("Requested width must match tensor width");
        for (int row : rows) {
            requireRow(tensor, row);
        }
        return Tensor.backend().sliceRows(tensor, rows);
    }

    static Tensor sampleRows(Tensor tensor, int count, Random random) {
        if (count < 1) throw new IllegalArgumentException("Sample count must be positive");
        tensor.materialize();
        return CPU.sampleRows(tensor, count, random);
    }

    static void print(Tensor tensor, String label) {
        tensor.materialize();
        CPU.print(tensor, label);
    }

    static Tensor zeros(int rows, int columns) { return CPU.zeros(rows, columns); }
    static Tensor ones(int rows, int columns) { return CPU.ones(rows, columns); }
    static Tensor random(int rows, int columns, Random random) { return CPU.random(rows, columns, random); }
    static Tensor causalMask(int size) { return CPU.causalMask(size); }

    static void requireSameShape(Tensor left, Tensor right, String operation) {
        if (left.rows == right.rows && left.cols == right.cols) return;
        throw new IllegalArgumentException("Shape mismatch for " + operation + ": "
                + left.rows + "x" + left.cols + " vs " + right.rows + "x" + right.cols);
    }

    static void requireTargetsMatchRows(Tensor logits, int[] targets) {
        if (targets.length == logits.rows) return;
        throw new IllegalArgumentException(
                "targets length " + targets.length + " must match logits rows " + logits.rows);
    }

    private static void requireIndex(Tensor tensor, int row, int column) {
        requireRow(tensor, row);
        if (column < 0 || column >= tensor.cols) throw new IndexOutOfBoundsException("Column index: " + column);
    }

    private static void requireRow(Tensor tensor, int row) {
        if (row < 0 || row >= tensor.rows) throw new IndexOutOfBoundsException("Row index: " + row);
    }

    private static void markGpuNeedsUpload(Tensor tensor) {
        if (!(tensor.gpuTag instanceof GpuBuffer buffer)) return;
        buffer.needsUpload = true;
        buffer.cpuStale = false;
    }
}
