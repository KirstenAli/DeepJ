package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.persistence.qwen3.SafeTensorWriter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/** Architecture and weights in a Hugging Face compatible directory, published only after both files are complete. */
public final class Qwen3Checkpoint {
    private Qwen3Checkpoint() {}

    /** The destination must not already exist; tokenizer files and optimizer state are managed separately. */
    public static void save(Qwen3 model, Path directory) throws IOException {
        Path destination = directory.toAbsolutePath();
        if (Files.exists(destination)) throw new java.nio.file.FileAlreadyExistsException(destination.toString());
        Files.createDirectories(destination.getParent());
        Path temporary = Files.createTempDirectory(destination.getParent(), ".qwen3-");
        try (var ignored = GradMode.noGrad(); var scope = TensorScope.open()) {
            Qwen3ConfigIO.write(temporary.resolve("config.json"), model.config());
            SafeTensorWriter.write(temporary.resolve("model.safetensors"), sourceWeights(model));
            Files.move(temporary, destination, StandardCopyOption.ATOMIC_MOVE);
        } finally {
            cleanup(temporary);
        }
    }

    public static Qwen3 load(Path directory) throws IOException { return Qwen3Pretrained.load(directory); }

    private static Map<String, Tensor> sourceWeights(Qwen3 model) {
        Map<String, Tensor> parameters = new HashMap<>(), result = new LinkedHashMap<>();
        model.parameters().forEach(parameter -> parameters.put(parameter.name(), parameter.value));
        for (var weight : Qwen3Weights.inventory(model.config())) {
            Tensor value = parameters.get(weight.name());
            result.put(weight.source(), weight.transpose() ? value.transpose() : value.reshape(weight.sourceShape()));
        }
        return result;
    }

    private static void cleanup(Path temporary) throws IOException {
        if (!Files.exists(temporary)) return;
        try (var files = Files.list(temporary)) {
            for (Path file : files.toList()) {
                Files.deleteIfExists(file);
            }
        }
        Files.deleteIfExists(temporary);
    }
}
