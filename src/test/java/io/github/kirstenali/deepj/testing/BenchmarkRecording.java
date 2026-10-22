package io.github.kirstenali.deepj.testing;

import jdk.jfr.Recording;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.time.Duration;

final class BenchmarkRecording implements AutoCloseable {
    private final Recording recording;
    private final Path destination;

    private BenchmarkRecording(Recording recording, Path destination) {
        this.recording = recording;
        this.destination = destination;
    }

    static BenchmarkRecording start() {
        String path = System.getProperty("deepj.benchmarkRecording");
        return start(path == null ? null : Path.of(path));
    }

    static BenchmarkRecording start(Path destination) {
        if (destination == null) return new BenchmarkRecording(null, null);
        Recording recording = configured();
        try {
            recording.start();
            return new BenchmarkRecording(recording, destination);
        } catch (RuntimeException exception) {
            recording.close();
            throw exception;
        }
    }

    private static Recording configured() {
        Recording recording = new Recording();
        recording.enable("jdk.ExecutionSample").withPeriod(Duration.ofMillis(10));
        recording.enable("jdk.NativeMethodSample").withPeriod(Duration.ofMillis(10));
        recording.enable("jdk.ObjectAllocationSample").with("throttle", "150/s");
        recording.enable("jdk.GarbageCollection").withoutStackTrace();
        recording.enable("jdk.GCPhasePause").withoutStackTrace();
        recording.enable("jdk.CPULoad").withPeriod(Duration.ofSeconds(1));
        recording.enable("jdk.ThreadCPULoad").withPeriod(Duration.ofSeconds(1));
        recording.enable(BenchmarkStep.class).withoutStackTrace();
        recording.setMaxSize(128L * 1_024 * 1_024);
        return recording;
    }

    @Override
    public void close() {
        if (recording == null) return;
        try {
            recording.stop();
            recording.dump(destination);
        } catch (IOException exception) {
            throw new UncheckedIOException(exception);
        } finally {
            recording.close();
        }
    }
}
