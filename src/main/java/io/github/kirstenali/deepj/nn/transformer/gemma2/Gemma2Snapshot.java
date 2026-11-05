package io.github.kirstenali.deepj.nn.transformer.gemma2;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/** Recognizes standalone Gemma 2 weights or the text_encoder folder of a local SANA snapshot. */
public final class Gemma2Snapshot {
    private Gemma2Snapshot() {}

    public static Path directory(Path path) {
        Path absolute = path.toAbsolutePath().normalize();
        return Files.isDirectory(absolute) ? absolute : absolute.getParent();
    }

    public static boolean matches(Path path) throws IOException {
        if (!Files.isDirectory(path) && !path.getFileName().toString()
                .matches("config\\.json|.+\\.safetensors(?:\\.index\\.json)?")) return false;
        Path config = directory(path).resolve("config.json");
        return Files.isRegularFile(config) && "gemma2".equals(Qwen3Json.read(config).get("model_type"));
    }

    public static Path tokenizer(Path directory) throws IOException {
        for (Path candidate : new Path[]{directory, directory.resolve("tokenizer"),
                directory.resolveSibling("tokenizer")}) {
            if (Files.isRegularFile(candidate.resolve("tokenizer.json"))) return candidate;
        }
        throw new IOException("Gemma tokenizer.json is missing");
    }
}
