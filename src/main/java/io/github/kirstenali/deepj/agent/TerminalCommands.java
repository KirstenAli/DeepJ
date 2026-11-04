package io.github.kirstenali.deepj.agent;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.regex.Pattern;

record TerminalCommands(Function<String, Path> lookup, String description, boolean inheritEnvironment) {
    static TerminalCommands allowlist(Map<String, Path> executables) {
        var allowed = Map.copyOf(executables);
        if (allowed.values().stream().anyMatch(path -> !path.isAbsolute())) {
            throw new IllegalArgumentException("Executable paths must be absolute");
        }
        String names = String.join(", ", allowed.keySet().stream().sorted().toList());
        String description = "Run an approved command in the working directory. Available commands: "
                + names + ". Supply its name first, followed by each argument separately.";
        return new TerminalCommands(allowed::get, description, false);
    }

    static TerminalCommands local(Path directory, String searchPath) {
        String path = Objects.requireNonNullElse(searchPath, "/usr/bin:/bin");
        var entries = Arrays.stream(path.split(Pattern.quote(File.pathSeparator), -1))
                .map(directory::resolve).toList();
        return new TerminalCommands(name -> resolve(directory, entries, name), """
                Run a local command after the user approves it. Provide the command name or path first, then
                each argument as a separate string. For pipes or redirects, explicitly request a shell such as
                ["/bin/sh", "-c", "your pipeline"].
                """.strip(), true);
    }

    private static Path resolve(Path directory, List<Path> entries, String name) {
        if (name.isBlank()) throw new IllegalArgumentException("argv must start with a command name or path");
        Path command = Path.of(name);
        if (name.indexOf(File.separatorChar) >= 0) {
            Path explicit = directory.resolve(command);
            return executable(explicit) ? explicit : null;
        }
        return entries.stream().map(entry -> entry.resolve(command)).filter(TerminalCommands::executable)
                .findFirst().orElse(null);
    }

    private static boolean executable(Path path) {
        return Files.isRegularFile(path) && Files.isExecutable(path);
    }
}
