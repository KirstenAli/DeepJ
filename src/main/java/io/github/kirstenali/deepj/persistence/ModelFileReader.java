package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.tensor.TensorDescriptor;
import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/** One bounded metadata scan shared by inspection and automatic model construction. */
final class ModelFileReader {
    private ModelFileReader() {}

    static Contents read(Path path) throws IOException {
        try (var file = new CheckpointInput(path)) {
            ModelFormat format = ModelFormat.read(file.data());
            if (format.settings() == null) throw new IOException("Model has no embedded settings: " + path);
            if (format.count() < 1) throw new IOException("Model has no tensors");
            Map<String, TensorDescriptor> descriptors = new LinkedHashMap<>();
            long elements = 0;
            for (int index = 0; index < format.count(); index++) {
                var entry = ModelTensorIO.descriptor(file.data(), format);
                add(descriptors, entry, format.named());
                file.skip(entry.tensor().bytes());
                elements += entry.tensor().elements();
            }
            if (file.position() != file.size()) throw new IOException("Unexpected trailing checkpoint data");
            return new Contents(format, Map.copyOf(descriptors), elements, file.size());
        }
    }

    private static void add(Map<String, TensorDescriptor> descriptors, ModelTensorIO.Entry entry,
                             boolean named) throws IOException {
        if (named && descriptors.putIfAbsent(entry.name(), entry.tensor()) != null) {
            throw new IOException("Duplicate checkpoint parameter name: " + entry.name());
        }
    }

    record Contents(ModelFormat format, Map<String, TensorDescriptor> parameters, long elements, long bytes) {
        ModelInspection inspection() {
            return new ModelInspection(format.settings(), elements, format.count(), bytes);
        }

        ModelLayout layout() throws IOException {
            if (format.typed() && !format.named()) {
                throw new IOException("Automatic typed model loading requires named parameter descriptors");
            }
            return new ModelLayout(format.settings(), parameters);
        }
    }
}
