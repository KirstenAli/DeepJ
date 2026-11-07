package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.nn.transformer.PromptFormat;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.persistence.ModelInspection;
import io.github.kirstenali.deepj.persistence.ModelSettings;
import io.github.kirstenali.deepj.persistence.qwen3.SafeTensorCatalog;
import io.github.kirstenali.deepj.persistence.qwen3.SafeTensorFile;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorDescriptor;
import io.github.kirstenali.deepj.tensor.TensorEncoding;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

/** Imports a local Hugging Face Qwen3 snapshot; no Python runtime or global loader changes are required. */
public final class Qwen3Pretrained {
    private Qwen3Pretrained() {}

    public static Qwen3 load(Path directory) throws IOException { return load(directory, null); }

    /** A null dtype preserves each source tensor's storage format. */
    public static Qwen3 load(Path directory, DType dtype) throws IOException {
        return load(directory, dtype, false);
    }

    /** Optionally retain compact activations as well as parameter storage. Gradients remain FLOAT32. */
    public static Qwen3 load(Path directory, DType dtype, boolean storageActivations) throws IOException {
        Layout layout = layout(directory, dtype);
        requireTiedHead(layout.source());
        ParameterFactory factory = new ParameterFactory(layout.descriptors());
        if (storageActivations) factory = factory.withStorageActivations();
        Qwen3 model = Qwen3.forLoading(layout.config(), factory);
        Map<String, Tensor> targets = new HashMap<>();
        model.parameters().forEach(parameter -> targets.put(parameter.name(), parameter.value));
        for (var weight : layout.weights()) {
            Tensor source = SafeTensorFile.read(layout.source().get(weight.source()));
            TensorEncoding.restoreWeight(targets.get(weight.name()), source, weight.transpose());
        }
        return model;
    }

    public static ModelInspection inspect(Path directory) throws IOException {
        Layout layout = layout(directory, null);
        var settings = new ModelSettings(layout.config(), PromptFormat.PLAIN,
                Map.of("name", "Qwen3", "prompt_format", "Qwen3 (non-thinking)"));
        long bytes = 0;
        for (Path file : layout.source().values().stream().map(SafeTensorFile.Entry::file).distinct().toList()) {
            bytes = Math.addExact(bytes, Files.size(file));
        }
        return new ModelInspection(settings, layout.config().parameterCount(), layout.weights().size(), bytes);
    }

    private static Layout layout(Path directory, DType dtype) throws IOException {
        Qwen3Config config = Qwen3ConfigIO.read(directory.resolve("config.json"));
        Map<String, SafeTensorFile.Entry> source = SafeTensorCatalog.read(directory);
        long expected = 11L * config.nLayers() + 2 + (source.containsKey("lm_head.weight") ? 1 : 0);
        if (source.size() != expected) throw new IOException("Qwen3 weight count does not match the configuration");
        List<Qwen3Weights.Weight> weights = Qwen3Weights.inventory(config);
        return new Layout(config, source, weights, validate(source, weights, dtype));
    }

    private record Layout(Qwen3Config config, Map<String, SafeTensorFile.Entry> source,
                          List<Qwen3Weights.Weight> weights, Map<String, TensorDescriptor> descriptors) {}

    private static Map<String, TensorDescriptor> validate(Map<String, SafeTensorFile.Entry> source,
            List<Qwen3Weights.Weight> weights, DType dtype) throws IOException {
        var remaining = new HashSet<>(source.keySet());
        remaining.remove("lm_head.weight");
        Map<String, TensorDescriptor> result = new HashMap<>();
        for (var weight : weights) {
            var entry = source.get(weight.source());
            if (entry == null || !Arrays.equals(entry.descriptor().shape(), weight.sourceShape())) {
                throw new IOException("Missing or incorrectly shaped weight: " + weight.source());
            }
            result.put(weight.name(), weight.descriptor(dtype == null ? entry.descriptor().dtype() : dtype));
            remaining.remove(weight.source());
        }
        if (!remaining.isEmpty()) throw new IOException("Unrecognized Qwen3 weights: " + remaining);
        return result;
    }

    private static void requireTiedHead(Map<String, SafeTensorFile.Entry> source) throws IOException {
        if (source.containsKey("lm_head.weight")) {
            SafeTensorFile.requireIdentical(source.get("model.embed_tokens.weight"), source.get("lm_head.weight"));
        }
    }

}
