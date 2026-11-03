package io.github.kirstenali.deepj.persistence.qwen3;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

final class JsonWriter {
    private JsonWriter() {}

    static String write(Object value) {
        if (value == null) return "null";
        if (value instanceof String text) return quote(text);
        if (value instanceof Boolean) return value.toString();
        if (value instanceof Number number) return number(number);
        if (value instanceof Map<?, ?> map) return object(map);
        if (value instanceof List<?> list) return list.stream().map(JsonWriter::write)
                .collect(Collectors.joining(", ", "[", "]"));
        throw new IllegalArgumentException("Unsupported JSON value");
    }

    private static String number(Number number) {
        if (!Double.isFinite(number.doubleValue())) throw new IllegalArgumentException("Non-finite JSON number");
        return number.toString();
    }

    private static String object(Map<?, ?> map) {
        return map.entrySet().stream().map(entry -> quote(Qwen3Json.string(entry.getKey()))
                + ": " + write(entry.getValue())).collect(Collectors.joining(", ", "{", "}"));
    }

    private static String quote(String text) {
        StringBuilder result = new StringBuilder("\"");
        for (int index = 0; index < text.length(); index++) {
            char value = text.charAt(index);
            if (value == '"' || value == '\\') result.append('\\').append(value);
            else if (value < 32 || Character.isSurrogate(value)) result.append(String.format("\\u%04x", (int) value));
            else result.append(value);
        }
        return result.append('"').toString();
    }
}
