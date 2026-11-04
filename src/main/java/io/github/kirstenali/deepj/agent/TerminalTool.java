package io.github.kirstenali.deepj.agent;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.function.Predicate;

/** Runs approved executables in a fixed working directory with bounded time and output. */
public final class TerminalTool implements Tool {
    private static final int OUTPUT_BYTES = 8192;
    private final Path directory;
    private final Map<String, Path> executables;
    private final Predicate<List<String>> authorize;
    private final long timeoutNanos;

    /**
     * The executable map binds tool-visible names to trusted absolute paths.
     * The policy receives an immutable argument list with the executable already resolved.
     */
    public TerminalTool(Path directory, Map<String, Path> executables,
                        Predicate<List<String>> authorize, Duration timeout) {
        this.directory = directory.toAbsolutePath();
        if (!Files.isDirectory(this.directory)) throw new IllegalArgumentException("Working directory must exist");
        this.executables = Map.copyOf(executables);
        if (this.executables.values().stream().anyMatch(path -> !path.isAbsolute())) {
            throw new IllegalArgumentException("Executable paths must be absolute");
        }
        this.authorize = Objects.requireNonNull(authorize, "authorize");
        timeoutNanos = timeout.toNanos();
        if (timeoutNanos <= 0) throw new IllegalArgumentException("Timeout must be positive");
    }

    @Override
    public String name() { return "terminal"; }

    @Override
    public String description() {
        return "Run an approved command in the working directory. Available commands: "
                + String.join(", ", executables.keySet().stream().sorted().toList())
                + ". Supply its name first, followed by each argument separately.";
    }

    @Override
    public Map<String, Object> parameters() {
        var argv = Map.of("type", "array", "items", Map.of("type", "string"), "minItems", 1);
        return Map.of("type", "object", "properties", Map.of("argv", argv),
                "required", List.of("argv"), "additionalProperties", false);
    }

    @Override
    public Map<String, Object> execute(Map<String, Object> arguments) throws IOException, InterruptedException {
        List<String> argv = arguments(arguments);
        Path executable = executables.get(argv.get(0));
        if (executable == null) return Map.of("error", "Command is not available");
        var command = new ArrayList<>(argv);
        command.set(0, executable.toString());
        if (!authorize.test(List.copyOf(command))) return Map.of("error", "Command denied by terminal policy");
        if (Thread.interrupted()) throw new InterruptedException("Terminal command interrupted");
        return run(command);
    }

    private static List<String> arguments(Map<String, Object> arguments) {
        if (!arguments.keySet().equals(Set.of("argv"))) {
            throw new IllegalArgumentException("Expected only the argv argument");
        }
        var argv = Qwen3Json.array(arguments.get("argv")).stream().map(Qwen3Json::string).toList();
        if (argv.isEmpty()) throw new IllegalArgumentException("argv must contain the command name");
        if (argv.stream().anyMatch(arg -> arg.indexOf(0) >= 0)) {
            throw new IllegalArgumentException("argv cannot contain NUL characters");
        }
        return argv;
    }

    private Map<String, Object> run(List<String> command) throws IOException, InterruptedException {
        var builder = new ProcessBuilder(command).directory(directory.toFile()).redirectErrorStream(true);
        builder.environment().clear();
        Process process = builder.start();
        long start = System.nanoTime();
        try {
            process.getOutputStream().close();
            var reader = capture(process);
            if (!process.waitFor(remaining(start), TimeUnit.NANOSECONDS)) return timedOut();
            return result(process.exitValue(), reader.get(remaining(start), TimeUnit.NANOSECONDS));
        } catch (TimeoutException failure) {
            return timedOut();
        } catch (ExecutionException failure) {
            throw new IOException("Cannot read command output", failure.getCause());
        } finally {
            if (process.isAlive()) terminate(process);
        }
    }

    private static FutureTask<byte[]> capture(Process process) {
        var task = new FutureTask<>(() -> {
            try (var input = process.getInputStream()) {
                byte[] output = input.readNBytes(OUTPUT_BYTES + 1);
                input.transferTo(OutputStream.nullOutputStream());
                return output;
            }
        });
        var reader = new Thread(task, "deepj-terminal-output");
        reader.setDaemon(true);
        reader.start();
        return task;
    }

    private long remaining(long start) {
        return Math.max(0, timeoutNanos - (System.nanoTime() - start));
    }

    private static void terminate(Process process) {
        try {
            process.descendants().forEach(ProcessHandle::destroyForcibly);
        } catch (RuntimeException ignored) {
            // Some sandboxes forbid enumerating processes.
        } finally {
            process.destroyForcibly();
        }
    }

    private static Map<String, Object> timedOut() {
        return Map.of("error", "Command timed out");
    }

    private static Map<String, Object> result(int exitCode, byte[] output) {
        String text = new String(output, 0, Math.min(output.length, OUTPUT_BYTES), StandardCharsets.UTF_8);
        return Map.of("exitCode", exitCode, "output", text, "truncated", output.length > OUTPUT_BYTES);
    }
}
