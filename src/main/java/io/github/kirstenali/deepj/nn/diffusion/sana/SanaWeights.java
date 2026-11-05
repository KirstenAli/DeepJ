package io.github.kirstenali.deepj.nn.diffusion.sana;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import io.github.kirstenali.deepj.persistence.qwen3.SafeTensorFile;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorEncoding;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** One pipeline stage owns its loaded parameters and releases their device retention on close. */
public final class SanaWeights implements AutoCloseable {
    private final Map<String, SafeTensorFile.Entry> entries;
    private final Map<String, Parameter> loaded = new LinkedHashMap<>();
    private final DType dtype;
    private boolean closed;

    private SanaWeights(Map<String, SafeTensorFile.Entry> entries, DType dtype) {
        this.entries = entries;
        this.dtype = dtype;
    }

    public static SanaWeights open(Path directory, DType dtype) throws IOException {
        Path index = directory.resolve("model.safetensors.index.json");
        var entries = Files.exists(index) ? shards(directory, index)
                : SanaTensorFile.inspect(singleFile(directory));
        return new SanaWeights(entries, java.util.Objects.requireNonNull(dtype));
    }

    /** Useful for small tensor fixtures and differentiable model experiments. */
    public static SanaWeights from(Map<String, Tensor> tensors) {
        var weights = new SanaWeights(Map.of(), DType.FLOAT32);
        tensors.forEach((name, tensor) -> weights.loaded.put(name, parameter(name, tensor)));
        return weights;
    }

    public Tensor tensor(String name) {
        if (closed) throw new IllegalStateException("Weights are closed");
        return loaded.computeIfAbsent(name, this::load).value;
    }

    public boolean contains(String name) { return loaded.containsKey(name) || entries.containsKey(name); }
    public List<Parameter> parameters() { return List.copyOf(loaded.values()); }

    private Parameter load(String name) {
        var entry = entries.get(name);
        if (entry == null) throw new IllegalArgumentException("Missing model weight: " + name);
        try {
            Tensor target = Tensor.backend().zeros(dtype, entry.descriptor().shape());
            TensorEncoding.restoreWeight(target, SafeTensorFile.read(entry), false);
            return parameter(name, target);
        } catch (IOException failure) {
            throw new UncheckedIOException(failure);
        }
    }

    private static Parameter parameter(String name, Tensor tensor) {
        try (var scope = TensorScope.open()) {
            var parameter = new Parameter(name, tensor);
            parameter.setGradient(null);
            return parameter;
        }
    }

    @Override public void close() {
        boolean populated = !loaded.isEmpty();
        loaded.values().forEach(p -> p.value.releaseDeviceBufferRetention());
        loaded.clear();
        closed = true;
        if (populated) Tensor.backend().releaseTemporaryResources();
    }

    private static Path singleFile(Path directory) {
        Path file = directory.resolve("diffusion_pytorch_model.safetensors");
        return Files.exists(file) ? file : directory.resolve("model.safetensors");
    }

    private static Map<String, SafeTensorFile.Entry> shards(Path directory, Path index) throws IOException {
        var weights = Qwen3Json.object(Qwen3Json.read(index).get("weight_map"));
        Map<String, SafeTensorFile.Entry> result = new HashMap<>();
        for (Object filename : new java.util.HashSet<>(weights.values())) {
            String name = Qwen3Json.string(filename);
            if (!name.matches("[A-Za-z0-9_-]+\\.safetensors")) throw new IOException("Invalid shard name");
            var entries = SanaTensorFile.inspect(directory.resolve(name));
            for (var entry : entries.entrySet()) {
                if (!filename.equals(weights.get(entry.getKey()))) throw new IOException("Shard index mismatch");
                if (result.put(entry.getKey(), entry.getValue()) != null) throw new IOException("Duplicate weight");
            }
        }
        if (!result.keySet().equals(weights.keySet())) throw new IOException("Missing indexed weight");
        return Map.copyOf(result);
    }
}
