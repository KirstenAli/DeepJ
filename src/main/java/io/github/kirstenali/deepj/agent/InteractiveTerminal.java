package io.github.kirstenali.deepj.agent;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;

import java.io.File;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Objects;
import java.util.function.Predicate;
import java.util.stream.Collectors;

/** Shared terminal policy for DeepJ's interactive agent launchers. */
public final class InteractiveTerminal {
    private final Path directory;
    private final Predicate<Request> approve;

    public InteractiveTerminal(Path directory, Predicate<Request> approve) {
        this.directory = directory.toAbsolutePath().normalize();
        this.approve = Objects.requireNonNull(approve, "approve");
    }

    public record Request(Path directory, List<String> argv) {
        public Request {
            argv = List.copyOf(argv);
        }

        public String preview() {
            return "Command and arguments: " + commandPreview()
                    + "\nWorking directory: " + directoryPreview();
        }

        public String commandPreview() { return displayJson(argv); }
        public String directoryPreview() { return displayJson(directory.toString()); }

        private static String displayJson(Object value) {
            return Qwen3Json.stringify(value).chars().mapToObj(character -> character >= 32 && character <= 126
                    ? Character.toString(character) : "\\u%04x".formatted(character)).collect(Collectors.joining());
        }
    }

    public static boolean available() {
        return File.separatorChar == '/';
    }

    public Agent agent(AgentModel model) {
        if (!available()) throw new IllegalStateException("Agent mode requires a Unix-style terminal");
        var terminal = TerminalTool.local(directory, argv -> approve.test(new Request(directory, argv)),
                Duration.ofMinutes(2));
        return new Agent(model, List.of(terminal), 4);
    }
}
