package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/** Identifies Qwen3 snapshots without loading weights or changing the selected backend. */
public final class Qwen3Snapshot {
    private Qwen3Snapshot() {}

    public static Path directory(Path selection) {
        Path path = selection.toAbsolutePath();
        return Files.isDirectory(path) ? path : path.getParent();
    }

    public static boolean matches(Path selection) throws IOException {
        if (!candidate(selection)) return false;
        Path config = directory(selection).resolve("config.json");
        return Files.isRegularFile(config) && "qwen3".equals(Qwen3Json.read(config).get("model_type"));
    }

    private static boolean candidate(Path path) {
        if (Files.isDirectory(path)) return true;
        String filename = path.getFileName().toString();
        return filename.equals("config.json") || filename.endsWith(".safetensors")
                || filename.endsWith(".safetensors.index.json");
    }
}
