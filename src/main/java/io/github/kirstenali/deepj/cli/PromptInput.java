package io.github.kirstenali.deepj.cli;

import picocli.CommandLine.Option;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

final class PromptInput {
    private static final int MAX_BYTES = 1024 * 1024;

    @Option(names = "--prompt", description = "Prompt text (preserves capitalization).")
    String prompt;

    @Option(names = "--prompt-file", description = "Read a UTF-8 prompt from this file.")
    Path file;

    String read(CliServices services) throws IOException {
        if (prompt != null && file != null) throw new IllegalArgumentException("Choose --prompt OR --prompt-file.");
        String value = prompt != null ? prompt : fromInput(services);
        if (value.isBlank()) throw new IllegalArgumentException("The prompt cannot be empty.");
        return value;
    }

    private String fromInput(CliServices services) throws IOException {
        if (file != null) {
            try (InputStream input = Files.newInputStream(file)) {
                return bounded(input);
            }
        }
        if (services.interactive()) throw new IllegalArgumentException("Provide --prompt, --prompt-file or pipe text.");
        return bounded(services.input());
    }

    private String bounded(InputStream input) throws IOException {
        byte[] bytes = input.readNBytes(MAX_BYTES + 1);
        if (bytes.length > MAX_BYTES) throw new IllegalArgumentException("Prompt input exceeds 1 MiB.");
        return new String(bytes, StandardCharsets.UTF_8);
    }
}
