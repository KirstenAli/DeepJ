package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.models.TransformerConfig;
import io.github.kirstenali.deepj.tensor.GpuBuffer;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.autograd.GradMode;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Objects;

public final class ModelSerializer {

    private static final int MAGIC = 0x444A4D44;
    public static final int CURRENT_FORMAT_VERSION = 3;

    private ModelSerializer() {}

    public static void save(List<Parameter> params, Path path) throws IOException {
        save(params, path, null);
    }

    public static void save(List<Parameter> params, Path path, ModelSettings settings)
            throws IOException {
        validateParams(params);
        ensureParentDirectory(path);
        Path temporary = temporaryPath(path);
        try {
            writeModel(params, temporary, settings);
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

    private static void writeParameterCount(DataOutputStream out, int count,
                                            ModelSettings settings) throws IOException {
        out.writeInt(MAGIC);
        out.writeInt(settings == null ? 1 : CURRENT_FORMAT_VERSION);
        if (settings != null) ModelSettingsIO.write(out, settings);
        out.writeInt(count);
    }

    private static void writeParameter(DataOutputStream out, Parameter p) throws IOException {
        writeTensor(out, p.value);
    }

    static void writeTensor(DataOutputStream out, Tensor t) throws IOException {
        t.materialize();
        writeTensorHeader(out, t);
        writeTensorData(out, t);
    }

    private static void writeTensorHeader(DataOutputStream out, Tensor t) throws IOException {
        out.writeInt(t.rows);
        out.writeInt(t.cols);
    }

    private static void writeTensorData(DataOutputStream out, Tensor t) throws IOException {
        for (float v : t.rawData()) {
            out.writeFloat(v);
        }
    }

    public static void load(List<Parameter> params, Path path) throws IOException {
        load(params, path, null);
    }

    public static ModelSettings load(List<Parameter> params, Path path,
                                     TransformerConfig expected) throws IOException {
        validateParams(params);
        prepareForLoad();
        try (DataInputStream in = openInput(path)) {
            Format format = readFormat(in);
            validateSettings(format.settings(), expected);
            int count = readAndValidateParameterCount(format.count(), params.size());
            for (int i = 0; i < count; i++) {
                Tensor t = params.get(i).value;
                readAndValidateShape(in, t, i);
                readTensorData(in, t);
                markGpuBufferNeedsUpload(t);
            }
            if (in.read() != -1) throw new IOException("Unexpected trailing checkpoint data");
            return format.settings();
        }
    }

    public static ModelSettings settings(Path path) throws IOException {
        try (DataInputStream input = openInput(path)) {
            return readFormat(input).settings();
        }
    }

    public static ModelInspection inspect(Path path) throws IOException {
        try (FileChannel channel = FileChannel.open(path);
             var input = new DataInputStream(Channels.newInputStream(channel))) {
            Format format = readFormat(input);
            if (format.settings() == null) throw new IOException("Model has no embedded settings: " + path);
            if (format.count() < 1) throw new IOException("Model has no tensors");
            long parameters = 0;
            for (int index = 0; index < format.count(); index++) {
                parameters += inspectTensor(channel, input);
            }
            if (channel.position() != channel.size()) throw new IOException("Unexpected trailing checkpoint data");
            return new ModelInspection(format.settings(), parameters, format.count(), channel.size());
        }
    }

    private static long inspectTensor(FileChannel channel, DataInputStream input) throws IOException {
        int rows = input.readInt(), columns = input.readInt();
        long count = (long) rows * columns;
        if (rows < 1 || columns < 1 || count > Integer.MAX_VALUE) {
            throw new IOException("Invalid checkpoint tensor shape");
        }
        long bytes = count * Float.BYTES;
        if (bytes > channel.size() - channel.position()) throw new IOException("Truncated checkpoint tensor");
        channel.position(channel.position() + bytes);
        return count;
    }

    private static void validateSettings(ModelSettings settings, TransformerConfig expected)
            throws IOException {
        if (settings != null && expected != null && !Objects.equals(settings.config(), expected)) {
            throw new IOException("Model configuration does not match the checkpoint");
        }
    }

    static void prepareForLoad() {
        Tensor.backend().releaseResources();
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

    static void readTensor(DataInputStream in, Tensor tensor, int index) throws IOException {
        readAndValidateShape(in, tensor, index);
        readTensorData(in, tensor);
    }

    private static void readTensorData(DataInputStream in, Tensor t) throws IOException {
        try (var ignored = GradMode.noGrad()) {
            t.gradientState().beforeMutation();
            float[] data = t.data();
            for (int j = 0; j < data.length; j++) {
                data[j] = in.readFloat();
            }
        }
    }

    static void markGpuBufferNeedsUpload(Tensor t) {

        if (t.getGpuTag() instanceof GpuBuffer gb) {
            gb.needsUpload = true;
            gb.cpuStale = false;
        }
    }

    private static void writeModel(List<Parameter> params, Path path, ModelSettings settings)
            throws IOException {
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(path)))) {
            writeParameterCount(out, params.size(), settings);
            for (Parameter parameter : params) {
                writeParameter(out, parameter);
            }
        }
    }

    private static DataInputStream openInput(Path path) throws IOException {
        return new DataInputStream(new BufferedInputStream(Files.newInputStream(path)));
    }

    private static Format readFormat(DataInputStream in) throws IOException {
        int marker = in.readInt();
        if (marker != MAGIC) throw new IOException("Unsupported checkpoint format: missing DeepJ header");
        int version = in.readInt();
        if (version == 1) return new Format(in.readInt(), null);
        if (version == CURRENT_FORMAT_VERSION) {
            ModelSettings settings = ModelSettingsIO.read(in);
            return new Format(in.readInt(), settings);
        }
        throw new IOException("Unsupported model format version: " + version);
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

    private record Format(int count, ModelSettings settings) {}
}
