package io.github.kirstenali.deepj.agent;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.function.Function;
import java.util.function.Predicate;

/** Runs approved executables in a fixed working directory with bounded time and output. */
public final class TerminalTool implements Tool {
    private static final int OUTPUT_BYTES = 8192;
    private final Path directory;
    private final TerminalCommands commands;
    private final Function<List<String>, Optional<List<String>>> review;
    private final long timeoutNanos;

    /**
     * The executable map binds tool-visible names to trusted absolute paths.
     * The policy receives an immutable argument list with the executable already resolved.
     */
    public TerminalTool(Path directory, Map<String, Path> executables,
                        Predicate<List<String>> authorize, Duration timeout) {
        this(directory, TerminalCommands.allowlist(executables), approval(authorize), timeout);
    }

    static TerminalTool local(Path directory, Predicate<List<String>> authorize, Duration timeout) {
        return localWithReview(directory, approval(authorize), timeout);
    }

    static TerminalTool localWithReview(Path directory, Function<List<String>, Optional<List<String>>> review,
                                       Duration timeout) {
        return new TerminalTool(directory, TerminalCommands.local(directory.toAbsolutePath(), System.getenv("PATH")),
                review, timeout);
    }

    private static Function<List<String>, Optional<List<String>>> approval(Predicate<List<String>> authorize) {
        Objects.requireNonNull(authorize, "authorize");
        return command -> authorize.test(command) ? Optional.of(command) : Optional.empty();
    }

    private TerminalTool(Path directory, TerminalCommands commands,
                         Function<List<String>, Optional<List<String>>> review, Duration timeout) {
        this.directory = directory.toAbsolutePath();
        if (!Files.isDirectory(this.directory)) throw new IllegalArgumentException("Working directory must exist");
        this.commands = commands;
        this.review = Objects.requireNonNull(review, "review");
        timeoutNanos = timeout.toNanos();
        if (timeoutNanos <= 0) throw new IllegalArgumentException("Timeout must be positive");
    }

    @Override
    public String name() { return "terminal"; }

    @Override
    public String description() {
        return commands.description();
    }

    @Override
    public Map<String, Object> parameters() {
        var argv = Map.of("type", "array", "items", Map.of("type", "string"), "minItems", 1);
        return Map.of("type", "object", "properties", Map.of("argv", argv),
                "required", List.of("argv"), "additionalProperties", false);
    }

    @Override
    public Map<String, Object> execute(Map<String, Object> arguments) throws IOException, InterruptedException {
        var proposed = resolve(commands, arguments(arguments));
        if (proposed == null) return Map.of("error", "Command is not available");
        var approved = review.apply(proposed).map(List::copyOf).orElse(null);
        if (approved == null) return Map.of("error", "Command denied by terminal policy");
        boolean edited = !approved.equals(proposed);
        if (edited && !approved.equals(resolve(commands, approved))) {
            return Map.of("error", "The reviewed executable is no longer available");
        }
        if (Thread.interrupted()) throw new InterruptedException("Terminal command interrupted");
        var result = run(approved);
        if (!edited) return result;
        var editedResult = new LinkedHashMap<>(result);
        editedResult.put("argv", approved);
        return Map.copyOf(editedResult);
    }

    static List<String> resolve(TerminalCommands commands, List<String> argv) {
        if (argv.isEmpty()) throw new IllegalArgumentException("argv must contain the command name");
        if (argv.stream().anyMatch(arg -> arg.indexOf(0) >= 0)) {
            throw new IllegalArgumentException("argv cannot contain NUL characters");
        }
        Path executable = commands.lookup().apply(argv.get(0));
        if (executable == null) return null;
        var resolved = new ArrayList<>(argv);
        resolved.set(0, executable.toString());
        return List.copyOf(resolved);
    }

    private static List<String> arguments(Map<String, Object> arguments) {
        if (!arguments.keySet().equals(Set.of("argv"))) {
            throw new IllegalArgumentException("Expected only the argv argument");
        }
        return Qwen3Json.array(arguments.get("argv")).stream().map(Qwen3Json::string).toList();
    }

    private Map<String, Object> run(List<String> command) throws IOException, InterruptedException {
        var builder = new ProcessBuilder(command).directory(directory.toFile()).redirectErrorStream(true);
        if (!commands.inheritEnvironment()) builder.environment().clear();
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
