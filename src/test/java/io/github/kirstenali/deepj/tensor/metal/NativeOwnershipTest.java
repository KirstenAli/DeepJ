package io.github.kirstenali.deepj.tensor.metal;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NativeOwnershipTest {

    private static final Path SOURCE = Path.of("native/deepj_metal_jni.mm");

    @Test
    void transferLookupDoesNotReturnMetalObject() throws IOException {
        String source = source();
        assertTrue(source.contains("static BufferEntry transferEntry"));
        assertFalse(source.contains("static id<MTLBuffer> transferBuffer"));
    }

    @Test
    void bufferBoundariesUseAutoreleasePools() throws IOException {
        String source = source();
        assertAutoreleasePool(source, "nativeAllocBuffers");
        assertAutoreleasePool(source, "nativeUploadBuffer");
        assertAutoreleasePool(source, "nativeDownloadBuffer");
        assertAutoreleasePool(source, "nativeReleaseBuffers");
    }

    private static void assertAutoreleasePool(String source, String function) {
        String expression = function + "\\s*\\([^)]*\\)\\s*\\{\\s*@autoreleasepool";
        assertTrue(Pattern.compile(expression).matcher(source).find(), function);
    }

    private static String source() throws IOException {
        return Files.readString(SOURCE);
    }
}
