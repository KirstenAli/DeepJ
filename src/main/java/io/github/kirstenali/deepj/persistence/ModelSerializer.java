package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.GpuBuffer;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public final class ModelSerializer {

    private ModelSerializer() {}

    public static void save(List<Parameter> params, Path path) throws IOException {
        validateParams(params);
        ensureParentDirectory(path);

        try (DataOutputStream out = new DataOutputStream(Files.newOutputStream(path))) {
            writeParameterCount(out, params.size());

            for (Parameter p : params) {
                writeParameter(out, p);
            }
        }
    }

    private static void validateParams(List<Parameter> params) {
        if (params == null) {
            throw new IllegalArgumentException("params is null");
        }
    }

    private static void ensureParentDirectory(Path path) throws IOException {
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
    }

    private static void writeParameterCount(DataOutputStream out, int count) throws IOException {
        out.writeInt(count);
    }

    private static void writeParameter(DataOutputStream out, Parameter p) throws IOException {
        Tensor t = p.value;
        t.materialize();
        writeTensorHeader(out, t);
        writeTensorData(out, t);
    }

    private static void writeTensorHeader(DataOutputStream out, Tensor t) throws IOException {
        out.writeInt(t.rows);
        out.writeInt(t.cols);
    }

    private static void writeTensorData(DataOutputStream out, Tensor t) throws IOException {
        for (double v : t.data) {
            out.writeDouble(v);
        }
    }

    public static void load(List<Parameter> params, Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(Files.newInputStream(path))) {
            int count = readAndValidateParameterCount(in, params.size());
            for (int i = 0; i < count; i++) {
                Tensor t = params.get(i).value;
                readAndValidateShape(in, t, i);
                readTensorData(in, t);
                markGpuBufferNeedsUpload(t);
            }
        }
    }

    private static int readAndValidateParameterCount(DataInputStream in, int expectedCount) throws IOException {
        int count = in.readInt();
        if (count != expectedCount) {
            throw new IOException("Parameter count mismatch");
        }
        return count;
    }

    private static void readAndValidateShape(DataInputStream in, Tensor t, int index) throws IOException {
        int rows = in.readInt();
        int cols = in.readInt();
        if (rows != t.rows || cols != t.cols) {
            throw new IOException("Shape mismatch at parameter " + index);
        }
    }

    private static void readTensorData(DataInputStream in, Tensor t) throws IOException {
        for (int j = 0; j < t.data.length; j++) {
            t.data[j] = (float) in.readDouble();
        }
    }

    private static void markGpuBufferNeedsUpload(Tensor t) {
        // If this tensor is already GPU-bound, force a fresh upload of loaded CPU weights.
        if (t.getGpuTag() instanceof GpuBuffer gb) {
            gb.needsUpload = true;
            gb.cpuStale = false;
        }
    }
}