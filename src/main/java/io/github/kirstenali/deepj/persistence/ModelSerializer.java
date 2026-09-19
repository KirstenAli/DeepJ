package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.GpuBuffer;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;

public final class ModelSerializer {

    private static final int MAGIC = 0x444A4D44;
    public static final int CURRENT_FORMAT_VERSION = 1;

    private ModelSerializer() {}

    public static void save(List<Parameter> params, Path path) throws IOException {
        validateParams(params);
        ensureParentDirectory(path);
        Path temporary = temporaryPath(path);
        try {
            writeModel(params, temporary);
            replaceAtomically(temporary, path);
        } finally {
            Files.deleteIfExists(temporary);
        }
    }

    private static void validateParams(List<Parameter> params) {
        if (params == null) throw new IllegalArgumentException("params is null");
        if (params.stream().anyMatch(p -> p == null || p.value == null)) {
            throw new IllegalArgumentException("params contains a null parameter or value");
        }
    }

    private static void ensureParentDirectory(Path path) throws IOException {
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
    }

    private static void writeParameterCount(DataOutputStream out, int count) throws IOException {
        out.writeInt(MAGIC);
        out.writeInt(CURRENT_FORMAT_VERSION);
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
        for (float v : t.data) {
            out.writeFloat(v);
        }
    }

    public static void load(List<Parameter> params, Path path) throws IOException {
        validateParams(params);
        try (DataInputStream in = openInput(path)) {
            Format format = readFormat(in);
            int count = readAndValidateParameterCount(format.count(), params.size());
            for (int i = 0; i < count; i++) {
                Tensor t = params.get(i).value;
                readAndValidateShape(in, t, i);
                readTensorData(in, t, format.legacy());
                markGpuBufferNeedsUpload(t);
            }
            if (in.read() != -1) throw new IOException("Unexpected trailing checkpoint data");
        }
    }

    private static int readAndValidateParameterCount(int count, int expectedCount) throws IOException {
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

    private static void readTensorData(DataInputStream in, Tensor t, boolean legacy) throws IOException {
        for (int j = 0; j < t.data.length; j++) {
            t.data[j] = legacy ? (float) in.readDouble() : in.readFloat();
        }
    }

    private static void markGpuBufferNeedsUpload(Tensor t) {

        if (t.getGpuTag() instanceof GpuBuffer gb) {
            gb.needsUpload = true;
            gb.cpuStale = false;
        }
    }

    private static void writeModel(List<Parameter> params, Path path) throws IOException {
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(path)))) {
            writeParameterCount(out, params.size());
            for (Parameter parameter : params) writeParameter(out, parameter);
        }
    }

    private static DataInputStream openInput(Path path) throws IOException {
        return new DataInputStream(new BufferedInputStream(Files.newInputStream(path)));
    }

    private static Format readFormat(DataInputStream in) throws IOException {
        int marker = in.readInt();
        if (marker != MAGIC) return new Format(marker, true);
        int version = in.readInt();
        if (version != CURRENT_FORMAT_VERSION) throw new IOException("Unsupported model format version: " + version);
        return new Format(in.readInt(), false);
    }

    private static Path temporaryPath(Path path) throws IOException {
        Path absolute = path.toAbsolutePath();
        return Files.createTempFile(absolute.getParent(), absolute.getFileName().toString(), ".tmp");
    }

    private static void replaceAtomically(Path temporary, Path destination) throws IOException {
        try {
            Files.move(temporary, destination, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(temporary, destination, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private record Format(int count, boolean legacy) {}
}
