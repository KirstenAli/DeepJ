package io.github.kirstenali.deepj.agent;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Predicate;

/** Shared terminal policy for DeepJ's interactive agent launchers. */
public final class InteractiveTerminal {
    private static final Map<String, Path> COMMANDS = Map.of("pwd", Path.of("/bin/pwd"), "ls", Path.of("/bin/ls"));
    private final Path directory;
    private final Predicate<Request> approve;

    public InteractiveTerminal(Path directory, Predicate<Request> approve) {
        this.directory = directory.toAbsolutePath().normalize();
        this.approve = Objects.requireNonNull(approve, "approve");
    }

    public record Request(Path directory, List<String> argv) {}

    public static boolean available() {
        return COMMANDS.values().stream().allMatch(path -> path.isAbsolute() && Files.isExecutable(path));
    }

    public Agent agent(AgentModel model) {
        if (!available()) throw new IllegalStateException("Agent mode requires /bin/pwd and /bin/ls");
        var terminal = new TerminalTool(directory, COMMANDS, this::approve, Duration.ofSeconds(5));
        return new Agent(model, List.of(terminal), 4);
    }

    private boolean approve(List<String> argv) {
        return argv.size() == 1 && approve.test(new Request(directory, argv));
    }
}
