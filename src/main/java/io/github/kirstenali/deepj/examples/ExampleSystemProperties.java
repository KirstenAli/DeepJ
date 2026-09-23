package io.github.kirstenali.deepj.examples;

import java.nio.file.Path;

final class ExampleSystemProperties {

    private ExampleSystemProperties() {}

    static int integer(String name, int fallback) {
        return Integer.parseInt(System.getProperty(name, Integer.toString(fallback)));
    }

    static float decimal(String name, float fallback) {
        return Float.parseFloat(System.getProperty(name, Float.toString(fallback)));
    }

    static long longValue(String name, long fallback) {
        return Long.parseLong(System.getProperty(name, Long.toString(fallback)));
    }

    static Path path(String name, String fallback) {
        return Path.of(System.getProperty(name, fallback));
    }

    static Path optionalPath(String name) {
        String value = System.getProperty(name);
        return value == null || value.isBlank() ? null : Path.of(value);
    }

    static Path nullablePath(String name) {
        String value = System.getProperty(name);
        return value == null ? null : Path.of(value);
    }
}
