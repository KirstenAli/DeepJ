package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorDescriptor;
import io.github.kirstenali.deepj.tensor.TensorEncoding;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;

/** Versioned parameter/buffer archives for a separately constructed network. Existing .dj formats are independent. */
public final class NetworkWeights {
    private static final int MAGIC = 0x444A4E4E;
    private static final int VERSION = 1;
    private NetworkWeights() {}

    public static void save(Path path, Layer model) throws IOException {
        Path target = path.toAbsolutePath(), temporary = Files.createTempFile(target.getParent(), ".deepj-", ".tmp");
        try {
            write(temporary, model);
            replace(temporary, target);
        } finally { Files.deleteIfExists(temporary); }
    }

    public static void load(Path path, Layer model) throws IOException {
        List<Entry> expected = entries(model);
        List<Tensor> values = read(path, model.getClass().getName(), expected);
        for (int index = 0; index < expected.size(); index++) {
            TensorEncoding.restore(expected.get(index).value(), values.get(index));
        }
        model.zeroGrad();
    }

    private static void write(Path path, Layer model) throws IOException {
        try (var output = new DataOutputStream(Files.newOutputStream(path))) {
            output.writeInt(MAGIC);
            output.writeInt(VERSION);
            output.writeUTF(model.getClass().getName());
            List<Entry> entries = entries(model);
            output.writeInt(entries.size());
            for (Entry entry : entries) {
                output.writeUTF(entry.name());
                TensorEncoding.write(output, entry.value());
            }
        }
    }

    private static List<Tensor> read(Path path, String type, List<Entry> expected) throws IOException {
        try (var input = new DataInputStream(Files.newInputStream(path))) {
            requireHeader(input, type, expected.size());
            List<Tensor> result = new ArrayList<>();
            for (Entry entry : expected) { result.add(readEntry(input, entry)); }
            if (input.read() != -1) throw new IOException("Unexpected trailing network weights data");
            return result;
        }
    }

    private static void requireHeader(DataInputStream input, String type, int size) throws IOException {
        if (input.readInt() != MAGIC || input.readInt() != VERSION) {
            throw new IOException("Invalid network weights format");
        }
        if (!input.readUTF().equals(type) || input.readInt() != size) {
            throw new IOException("Network weights do not match the constructed model");
        }
    }

    private static Tensor readEntry(DataInputStream input, Entry expected) throws IOException {
        if (!input.readUTF().equals(expected.name())) throw new IOException("Network parameter name mismatch");
        TensorDescriptor descriptor = TensorDescriptor.read(input);
        descriptor.requireMatches(expected.value());
        Tensor value = new Tensor(descriptor.dtype(), descriptor.shape());
        TensorEncoding.readValues(input, value);
        return value;
    }

    private static List<Entry> entries(Layer model) {
        List<Entry> result = new ArrayList<>();
        var seen = new IdentityHashMap<Tensor, Boolean>();
        List<Parameter> parameters = model.parameters();
        for (int index = 0; index < parameters.size(); index++) {
            Parameter parameter = parameters.get(index);
            add(result, seen, "parameter." + index + "." + parameter.name(), parameter.value);
        }
        if (model instanceof Module module) addBuffers(result, seen, module.buffers());
        return List.copyOf(result);
    }

    private static void addBuffers(List<Entry> result, IdentityHashMap<Tensor, Boolean> seen, List<Tensor> buffers) {
        for (int index = 0; index < buffers.size(); index++) {
            add(result, seen, "buffer." + index, buffers.get(index));
        }
    }

    private static void add(List<Entry> result, IdentityHashMap<Tensor, Boolean> seen, String name, Tensor tensor) {
        if (seen.put(tensor, Boolean.TRUE) == null) result.add(new Entry(name, tensor));
    }

    private static void replace(Path temporary, Path target) throws IOException {
        try {
            Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (AtomicMoveNotSupportedException error) {
            Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private record Entry(String name, Tensor value) {}
}
