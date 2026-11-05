package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

public final class HuggingFaceConfig {
    private final Map<String, Object> values;

    public HuggingFaceConfig(Path path) throws IOException { values = Qwen3Json.read(path); }
    public int integer(String name) { return Math.toIntExact(Qwen3Json.integer(required(name))); }
    public float decimal(String name) { return ((Number) required(name)).floatValue(); }
    public boolean flag(String name) { return Boolean.TRUE.equals(required(name)); }
    public String text(String name) { return Qwen3Json.string(required(name)); }
    public List<?> array(String name) { return Qwen3Json.array(required(name)); }
    public boolean isNull(String name) { return values.get(name) == null; }

    public int[] integers(String name) {
        return array(name).stream().mapToInt(v -> Math.toIntExact(Qwen3Json.integer(v))).toArray();
    }

    public void require(String name, Object expected) {
        if (!expected.equals(required(name))) throw new IllegalArgumentException("Unsupported " + name);
    }

    private Object required(String name) {
        if (!values.containsKey(name)) throw new IllegalArgumentException("Missing config field: " + name);
        return values.get(name);
    }
}
