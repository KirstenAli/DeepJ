package io.github.kirstenali.deepj.examples;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class ExampleSystemPropertiesTest {

    private static final String PROPERTY = "deepj.test.path";

    @AfterEach
    void clearProperty() {
        System.clearProperty(PROPERTY);
    }

    @Test
    void nullablePathReturnsNullWhenPropertyIsAbsent() {
        assertNull(ExampleSystemProperties.nullablePath(PROPERTY));
    }

    @Test
    void nullablePathPreservesBlankPropertyAsEmptyPath() {
        System.setProperty(PROPERTY, "");
        assertEquals(Path.of(""), ExampleSystemProperties.nullablePath(PROPERTY));
    }
}
