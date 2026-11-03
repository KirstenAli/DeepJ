package io.github.kirstenali.deepj.persistence.qwen3;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

/** A single local file or an indexed set of local safetensors shards. */
public final class SafeTensorCatalog {
    private SafeTensorCatalog() {}

    public static Map<String, SafeTensorFile.Entry> read(Path directory) throws IOException {
        Path index = directory.resolve("model.safetensors.index.json");
        if (!Files.exists(index)) return SafeTensorFile.inspect(directory.resolve("model.safetensors"));
        try { return shards(directory.toRealPath(), Qwen3Json.object(Qwen3Json.read(index).get("weight_map"))); }
        catch (IllegalArgumentException | ArithmeticException failure) {
            throw new IOException("Invalid safetensors shard index", failure);
        }
    }

    private static Map<String, SafeTensorFile.Entry> shards(Path directory, Map<String, Object> weights)
            throws IOException {
        Map<String, SafeTensorFile.Entry> result = new HashMap<>();
        var filenames = new HashSet<>(weights.values());
        for (Object filename : filenames) {
            Path path = localShard(directory, Qwen3Json.string(filename));
            for (var entry : SafeTensorFile.inspect(path).entrySet()) {
                if (!filename.equals(weights.get(entry.getKey()))
                        || result.put(entry.getKey(), entry.getValue()) != null) {
                    throw new IOException("Shard contents disagree with the weight index: " + entry.getKey());
                }
            }
        }
        if (!result.keySet().equals(weights.keySet())) throw new IOException("Missing indexed weights");
        return Map.copyOf(result);
    }

    private static Path localShard(Path directory, String filename) throws IOException {
        if (!filename.matches("[A-Za-z0-9_.-]+\\.safetensors") || filename.contains("..")) {
            throw new IOException("Invalid shard filename: " + filename);
        }
        Path path = directory.resolve(filename).toRealPath();
        if (!directory.equals(path.getParent())) throw new IOException("Shard is outside the model directory");
        return path;
    }
}
