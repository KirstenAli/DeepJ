package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import java.io.DataInput;
import java.io.IOException;
import java.nio.ByteOrder;

/** Immutable Q8_0 rows. Each group of 32 signed bytes shares an FP16 scale. */
public final class PackedQ8 implements AutoCloseable {
    private final Tensor storage;
    private final TensorBackend backend;
    private final int rows;
    private final int width;
    private boolean closed;

    private PackedQ8(Tensor storage, int rows, int width) {
        this.storage = storage;
        this.backend = Tensor.backend();
        this.rows = rows;
        this.width = width;
    }

    public static PackedQ8 read(DataInput source, int rows, int width) throws IOException {
        if (rows <= 0 || width <= 0 || width % 32 != 0) throw new IllegalArgumentException("Invalid Q8_0 shape");
        int words = Math.multiplyExact(rows, width / 32 * 17);
        TensorBackend backend = Tensor.backend();
        if (backend.usesDeviceStorage() && !(backend instanceof PackedQ8Backend)) {
            throw new IllegalStateException("The selected backend does not implement Q8_0");
        }
        Tensor buffer = backend.usesDeviceStorage()
                ? TensorEncoding.readValuesToBackend(source, ByteOrder.LITTLE_ENDIAN, DType.FLOAT16, 1, words)
                : new Tensor(DType.FLOAT16, 1, words);
        if (!backend.usesDeviceStorage()) TensorEncoding.readValues(source, buffer, ByteOrder.LITTLE_ENDIAN);
        return new PackedQ8(buffer.retainDeviceBuffer(), rows, width);
    }

    public Tensor project(Tensor input) {
        requireBackend();
        if (input.rank() < 2 || input.size(-1) != width) throw new IllegalArgumentException("Q8_0 width mismatch");
        int[] shape = input.shape();
        shape[shape.length - 1] = rows;
        Tensor matrix = input.reshape(input.numel() / width, width);
        Tensor result = backend instanceof PackedQ8Backend accelerated
                ? accelerated.projectQ8(matrix, storage, rows, width) : projectCpu(matrix);
        return result.reshape(shape);
    }

    public Tensor gather(int[] indices) {
        requireBackend();
        for (int row : indices) {
            if (row < 0 || row >= rows) throw new IllegalArgumentException("Invalid embedding index: " + row);
        }
        if (backend instanceof PackedQ8Backend accelerated) return accelerated.gatherQ8(storage, rows, width, indices);
        float[] output = new float[Math.multiplyExact(width, indices.length)];
        short[] bits = storage.memory().shorts();
        for (int i = 0; i < indices.length; i++) {
            Q8Reference.expand(bits, indices[i], width, output, i * width);
        }
        return Tensor.from(output, indices.length, width);
    }

    private Tensor projectCpu(Tensor matrix) {
        float[] input = matrix.toFloatArray();
        int batch = matrix.size(0);
        Tensor output = new Tensor(batch, rows);
        short[] bits = storage.memory().shorts();
        if (CpuBlas.AVAILABLE && batch > 0) {
            int workers = DeepJExecutor.isParallelEnabled() ? DeepJExecutor.getNumThreads() : 1;
            CpuBlas.q8Project(bits, input, output.rawData(), rows, width, batch, workers);
        } else {
            Q8Reference.project(bits, input, output.rawData(), rows, width);
        }
        return output;
    }

    private void requireBackend() {
        if (closed) throw new IllegalStateException("Q8_0 weights are closed");
        if (GradMode.isEnabled()) throw new IllegalStateException("Q8_0 projections require inference mode");
        if (backend != Tensor.backend()) throw new IllegalStateException("Q8_0 weights belong to another backend");
    }

    @Override public void close() {
        if (closed) return;
        closed = true;
        TensorEncoding.releaseOwnedWeight(storage);
    }
}
