package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.DType;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/** Validates local pipeline stages without loading their parameter payloads. */
public final class SanaVideoSnapshot {
    private SanaVideoSnapshot() {}

    public static boolean matches(Path selection) {
        try { return locate(selection) != null; }
        catch (IOException | IllegalArgumentException invalid) { return false; }
    }

    public static Path directory(Path selection) throws IOException {
        Path directory = locate(selection);
        if (directory == null) throw new IOException("Choose the SANA-Video folder containing model_index.json");
        validate(directory);
        return directory;
    }

    private static Path locate(Path selection) throws IOException {
        Path candidate = selection.toAbsolutePath().normalize();
        if (!Files.isDirectory(candidate)) candidate = candidate.getParent();
        for (int level = 0; level < 3 && candidate != null; level++, candidate = candidate.getParent()) {
            Path index = candidate.resolve("model_index.json");
            if (Files.isRegularFile(index)
                    && "SanaVideoPipeline".equals(Qwen3Json.read(index).get("_class_name"))) return candidate;
        }
        return null;
    }

    public static void validate(Path directory) throws IOException {
        if (!"SanaVideoPipeline".equals(Qwen3Json.read(directory.resolve("model_index.json")).get("_class_name"))) {
            throw new IOException("Expected a SANA-Video pipeline folder");
        }
        for (String file : new String[]{"tokenizer/tokenizer.json", "scheduler/scheduler_config.json",
                "text_encoder/config.json", "transformer/config.json", "vae/config.json"}) {
            if (!Files.isRegularFile(directory.resolve(file))) throw new IOException("Missing video file: " + file);
        }
        prepareIndex(directory.resolve("transformer"));
        for (String stage : new String[]{"text_encoder", "transformer", "vae"}) {
            try (var ignored = TensorWeights.open(directory.resolve(stage), DType.FLOAT32)) {
                // Metadata validation only; each network stage subsequently loads its own parameters.
            }
        }
    }

    private static void prepareIndex(Path stage) throws IOException {
        Path target = stage.resolve("model.safetensors.index.json");
        Path official = stage.resolve("diffusion_pytorch_model.safetensors.index.json");
        if (Files.exists(target) || !Files.exists(official)) return;
        try { Files.copy(official, target); }
        catch (java.nio.file.FileAlreadyExistsException concurrent) {
            // Another caller prepared the same immutable index. TensorWeights validates its contents.
        }
    }
}
