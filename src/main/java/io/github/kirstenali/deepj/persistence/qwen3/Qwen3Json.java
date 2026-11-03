package io.github.kirstenali.deepj.persistence.qwen3;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/** Dependency-free JSON input for Qwen configuration, tokenizer and safetensors metadata. */
public final class Qwen3Json {
    private Qwen3Json() {}

    public static Map<String, Object> read(Path path) throws IOException {
        if (Files.size(path) > 64L * 1024 * 1024) throw new IOException("JSON file exceeds 64 MiB");
        try { return object(parse(Files.readString(path))); }
        catch (IllegalArgumentException failure) { throw new IOException("Invalid JSON: " + path, failure); }
    }

    public static Object parse(String text) { return new JsonReader(text).read(); }
    public static String stringify(Object value) { return JsonWriter.write(value); }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> object(Object value) {
        if (!(value instanceof Map<?, ?>)) throw new IllegalArgumentException("Expected a JSON object");
        return (Map<String, Object>) value;
    }

    public static List<?> array(Object value) {
        if (!(value instanceof List<?> result)) throw new IllegalArgumentException("Expected a JSON array");
        return result;
    }

    public static String string(Object value) {
        if (!(value instanceof String result)) throw new IllegalArgumentException("Expected a JSON string");
        return result;
    }

    public static long integer(Object value) {
        if (!(value instanceof Long result)) throw new IllegalArgumentException("Expected a JSON integer");
        return result;
    }

    public static int integer(Map<String, Object> object, String key) {
        return Math.toIntExact(integer(object.get(key)));
    }

    public static double number(Object value) {
        if (!(value instanceof Number result)) throw new IllegalArgumentException("Expected a JSON number");
        return result.doubleValue();
    }
}
