package io.github.kirstenali.deepj.nn.diffusion.sana;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/** Resolves a local SANA pipeline folder or a file selected inside one of its model stages. */
public final class SanaSnapshot {
    private SanaSnapshot() {}

    public static boolean matches(Path selection) {
        try {
            directory(selection);
            return true;
        }
        catch (IOException | IllegalArgumentException invalid) { return false; }
    }

    public static Path directory(Path selection) throws IOException {
        Path candidate = selection.toAbsolutePath().normalize();
        if (!Files.isDirectory(candidate)) candidate = candidate.getParent();
        for (int level = 0; level < 3 && candidate != null; level++, candidate = candidate.getParent()) {
            Path index = candidate.resolve("model_index.json");
            if (Files.isRegularFile(index)
                    && "SanaSprintPipeline".equals(Qwen3Json.read(index).get("_class_name"))) {
                validate(candidate);
                return candidate;
            }
        }
        throw new IOException("Choose the SANA-Sprint folder containing model_index.json");
    }

    private static void validate(Path directory) throws IOException {
        for (String name : new String[]{"tokenizer/tokenizer.json", "text_encoder/config.json",
                "transformer/config.json", "vae/config.json"}) {
            if (!Files.isRegularFile(directory.resolve(name))) throw new IOException("Missing SANA file: " + name);
        }
        for (String stage : new String[]{"text_encoder", "transformer", "vae"}) {
            try (var ignored = SanaWeights.open(directory.resolve(stage),
                    io.github.kirstenali.deepj.tensor.DType.FLOAT32)) {
                // Validate metadata without loading weights.
            }
        }
    }
}
