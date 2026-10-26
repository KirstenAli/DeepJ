package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.autograd.LayoutOps;

import java.util.Arrays;
import java.util.Random;

final class TensorStorage {
    private static final CpuBackend CPU = new CpuBackend();

    private TensorStorage() {}

    static void discardStaleHostStorage(Tensor tensor) {
        tensor.memory().discardStaleHost();
    }

    static TensorBackend requireBackend(TensorBackend backend) {
        if (backend == null) throw new IllegalArgumentException("backend cannot be null");
        return backend;
    }

    static float[] hostStorage(Tensor tensor) {
        return tensor.memory().floats();
    }

    static float[] hostData(Tensor tensor) {
        materialize(tensor);
        return tensor.rawData();
    }

    static float item(Tensor tensor) {
        return TensorValues.item(tensor);
    }

    static int checkedSize(int rows, int cols) {
        if (rows < 1 || cols < 1) throw new IllegalArgumentException("Tensor dimensions must be positive");
        try {
            return Math.multiplyExact(rows, cols);
        } catch (ArithmeticException error) {
            throw new IllegalArgumentException("Tensor shape is too large: " + rows + "x" + cols, error);
        }
    }

    static TensorMemory copyMemory(Tensor source) {
        requireSource(source);
        return TensorValues.copy(source);
    }

    static void requireSource(Tensor source) {
        if (source == null) throw new IllegalArgumentException("source cannot be null");
    }

    static float[] rowData(Tensor tensor, int row) {
        requireRow(tensor, row);
        tensor.materialize();
        return Arrays.copyOfRange(tensor.rawData(), row * tensor.matrixCols(), (row + 1) * tensor.matrixCols());
    }

    static void materialize(Tensor tensor) {
        if (tensor.getGpuTag() != null) Tensor.backend().materializeTensor(tensor);
        if (tensor.getGpuTag() instanceof GpuBuffer buffer && buffer.cpuStale && buffer.owner != null) {
            buffer.owner.materialize(tensor);
        }
        tensor.memory().ensureHost();
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
            requireRowWidth(data[row], tensor.matrixCols());
            System.arraycopy(data[row], 0, tensor.rawData(), row * tensor.matrixCols(), tensor.matrixCols());
        }
    }

    private static void requireRowWidth(float[] row, int columns) {
        if (row == null || row.length != columns) {
            throw new IllegalArgumentException("All rows must have the same length (expected " + columns + ")");
        }
    }

    static Tensor getRow(Tensor tensor, int row) {
        requireRow(tensor, row);
        tensor.materialize();
        return CPU.getRow(tensor, row);
    }

    static void setRow(Tensor tensor, int row, Tensor source, int sourceRow) {
        requireRow(tensor, row);
        requireRow(source, sourceRow);
        if (source.matrixCols() != tensor.matrixCols()) {
            throw new IllegalArgumentException("Source row width must match tensor width");
        }
        tensor.materialize();
        source.materialize();
        CPU.setRow(tensor, row, source, sourceRow);
        markGpuNeedsUpload(tensor);
    }

    static Tensor sliceRows(Tensor tensor, int[] rows, int columns) {
        if (columns != tensor.matrixCols()) {
            throw new IllegalArgumentException("Requested width must match tensor width");
        }
        for (int row : rows) {
            requireRow(tensor, row);
        }
        return LayoutOps.gather(tensor, rows);
    }

    static Tensor sampleRows(Tensor tensor, int count, Random random) {
        if (count < 1) throw new IllegalArgumentException("Sample count must be positive");
        int[] indices = new int[count];
        for (int index = 0; index < count; index++) {
            indices[index] = random.nextInt(tensor.matrixRows());
        }
        return sliceRows(tensor, indices, tensor.matrixCols());
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
        if (left.layout().shape().equals(right.layout().shape())) return;
        throw new IllegalArgumentException("Shape mismatch for " + operation + ": "
                + left.layout().shape() + " vs " + right.layout().shape());
    }

    static void requireTargetsMatchRows(Tensor logits, int[] targets) {
        if (targets.length == logits.matrixRows()) return;
        throw new IllegalArgumentException(
                "targets length " + targets.length + " must match logits rows " + logits.matrixRows());
    }

    private static void requireRow(Tensor tensor, int row) {
        if (row < 0 || row >= tensor.matrixRows()) throw new IndexOutOfBoundsException("Row index: " + row);
    }

    static void markGpuNeedsUpload(TensorStorageAccess tensor) {
        if (!(tensor.getGpuTag() instanceof GpuBuffer buffer)) return;
        buffer.needsUpload = true;
        buffer.cpuStale = false;
    }
}
