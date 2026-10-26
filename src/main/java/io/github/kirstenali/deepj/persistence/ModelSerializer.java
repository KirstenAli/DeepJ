package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.models.TransformerConfig;
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
import java.util.Objects;

public final class ModelSerializer {

    public static final int CURRENT_FORMAT_VERSION = ModelFormat.CURRENT;

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

    public static void load(List<Parameter> params, Path path) throws IOException {
        load(params, path, null);
    }

    public static ModelSettings load(List<Parameter> params, Path path,
                                     TransformerConfig expected) throws IOException {
        validateParams(params);
        try (var file = new CheckpointInput(path)) {
            ModelFormat format = validateFile(file, params, expected);
            prepareForLoad();
            file.rewind();
            DataInputStream input = file.data();
            ModelFormat.read(input);
            for (Parameter parameter : params) {
                CheckpointTensorIO.readInto(input, parameter.value, format.typed());
            }
            if (input.read() != -1) throw new IOException("Unexpected trailing checkpoint data");
            return format.settings();
        }
    }

    public static ModelSettings settings(Path path) throws IOException {
        try (DataInputStream input = openInput(path)) {
            return ModelFormat.read(input).settings();
        }
    }

    public static ModelInspection inspect(Path path) throws IOException {
        try (var file = new CheckpointInput(path)) {
            ModelFormat format = ModelFormat.read(file.data());
            if (format.settings() == null) throw new IOException("Model has no embedded settings: " + path);
            if (format.count() < 1) throw new IOException("Model has no tensors");
            long parameters = 0;
            for (int index = 0; index < format.count(); index++) {
                parameters += CheckpointTensorIO.inspect(file, format.typed()).elements();
            }
            if (file.position() != file.size()) throw new IOException("Unexpected trailing checkpoint data");
            return new ModelInspection(format.settings(), parameters, format.count(), file.size());
        }
    }

    private static ModelFormat validateFile(CheckpointInput file, List<Parameter> params,
                                            TransformerConfig expected) throws IOException {
        ModelFormat format = ModelFormat.read(file.data());
        validateSettings(format.settings(), expected);
        if (format.count() != params.size()) throw new IOException("Parameter count mismatch");
        for (Parameter parameter : params) {
            var descriptor = CheckpointTensorIO.descriptor(file.data(), format.typed());
            descriptor.requireMatches(parameter.value);
            file.skip(descriptor.bytes());
        }
        if (file.position() != file.size()) throw new IOException("Unexpected trailing checkpoint data");
        return format;
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

    private static void writeModel(List<Parameter> params, Path path, ModelSettings settings)
            throws IOException {
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(path)))) {
            boolean typed = params.stream().anyMatch(parameter -> CheckpointTensorIO.requiresTyped(parameter.value));
            ModelFormat.write(out, params.size(), settings, typed);
            for (Parameter parameter : params) {
                CheckpointTensorIO.write(out, parameter.value, typed);
            }
        }
    }

    private static DataInputStream openInput(Path path) throws IOException {
        return new DataInputStream(new BufferedInputStream(Files.newInputStream(path)));
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

}
