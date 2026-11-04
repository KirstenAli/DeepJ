package io.github.kirstenali.deepj.agent;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;

import java.io.File;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;

/** Shared terminal policy for DeepJ's interactive agent launchers. */
public final class InteractiveTerminal {
    private final Path directory;
    private final Function<Request, Optional<Request>> review;

    public InteractiveTerminal(Path directory, Predicate<Request> approve) {
        this(directory, approval(approve));
    }

    /** Lets an interactive reviewer deny, approve, or approve a revised command. */
    public static InteractiveTerminal withCommandReview(Path directory, Function<Request, Optional<Request>> review) {
        return new InteractiveTerminal(directory, review);
    }

    private InteractiveTerminal(Path directory, Function<Request, Optional<Request>> review) {
        this.directory = directory.toAbsolutePath().normalize();
        this.review = Objects.requireNonNull(review, "review");
    }

    private static Function<Request, Optional<Request>> approval(Predicate<Request> approve) {
        Objects.requireNonNull(approve, "approve");
        return request -> approve.test(request) ? Optional.of(request) : Optional.empty();
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

        /** Resolves an edited command for review in the same working directory. */
        public Request revise(List<String> replacement) {
            var commands = TerminalCommands.local(directory.toAbsolutePath(), System.getenv("PATH"));
            var resolved = TerminalTool.resolve(commands, replacement);
            if (resolved == null) throw new IllegalArgumentException("The executable could not be found.");
            return new Request(directory, resolved);
        }

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
        var terminal = TerminalTool.localWithReview(directory, this::reviewCommand,
                Duration.ofMinutes(2));
        return new Agent(model, List.of(terminal), 4);
    }

    private Optional<List<String>> reviewCommand(List<String> argv) {
        return review.apply(new Request(directory, argv)).map(request -> {
            if (!directory.equals(request.directory())) throw new IllegalArgumentException("Working directory changed");
            return request.argv();
        });
    }
}
