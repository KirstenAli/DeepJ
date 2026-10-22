package io.github.kirstenali.deepj.testing;

import jdk.jfr.consumer.RecordingFile;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class BenchmarkRecordingTest {
    @TempDir Path directory;

    @Test
    void disabledRecordingDoesNotCreateAFile() {
        try (var ignored = BenchmarkRecording.start((Path) null)) {
            BenchmarkStep.start("measured", 0).finish();
        }
        assertEquals(0, directory.toFile().list().length);
    }

    @Test
    void recordsStepEventsWithoutEnvironmentOrSystemProperties() throws Exception {
        Path path = directory.resolve("trace.jfr");
        try (var ignored = BenchmarkRecording.start(path)) {
            BenchmarkStep.start("measured", 3).finish();
        }
        var events = RecordingFile.readAllEvents(path);
        var step = events.stream().filter(e -> e.getEventType().getName().equals("deepj.BenchmarkStep"))
                .findFirst().orElseThrow();
        assertEquals("measured", step.getString("phase"));
        assertEquals(3, step.getInt("round"));
        Set<String> sensitive = Set.of("jdk.InitialEnvironmentVariable", "jdk.InitialSystemProperty",
                "jdk.SystemProcess", "jdk.JVMInformation");
        assertFalse(events.stream().anyMatch(e -> sensitive.contains(e.getEventType().getName())));
        assertTrue(Files.size(path) > 0);
    }

    @Test
    void invalidDestinationFailsInsteadOfSilentlyDroppingTheProfile() {
        assertThrows(UncheckedIOException.class, () -> {
            try (var ignored = BenchmarkRecording.start(directory.resolve("missing/trace.jfr"))) {
                BenchmarkStep.start("measured", 0).finish();
            }
        });
    }
}
