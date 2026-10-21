package io.github.kirstenali.deepj.tensor.metal;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NativeSourceFilesTest {
    @TempDir Path directory;

    @Test
    void productionAndNativeTestHeadersAreIncluded() throws IOException {
        List<Path> files = NativeSourceFiles.all();
        assertTrue(files.contains(Path.of("native/deepj_metal_jni.mm")));
        assertTrue(files.contains(Path.of("src/test/native/coverage-hooks.h")));
        assertTrue(files.contains(Path.of("src/test/native/metal-analysis-shim.h")));
    }

    @ParameterizedTest
    @ValueSource(strings = {"h", "hpp", "c", "cc", "cpp", "m", "mm", "metal"})
    void nestedSourcesAreDiscovered(String extension) throws IOException {
        Path nested = Files.createDirectories(directory.resolve("tests/helpers/nested"));
        Path source = Files.writeString(nested.resolve("helper." + extension), "");
        assertEquals(List.of(source), NativeSourceFiles.under(List.of(directory)));
    }

    @Test
    void binariesAndJavaScriptFixturesAreNotNativeSource() throws IOException {
        Files.writeString(directory.resolve("library.dylib"), "");
        Files.writeString(directory.resolve("fixtures.mjs"), "");
        Files.writeString(directory.resolve("h"), "");
        assertTrue(NativeSourceFiles.under(List.of(directory)).isEmpty());
    }
}
