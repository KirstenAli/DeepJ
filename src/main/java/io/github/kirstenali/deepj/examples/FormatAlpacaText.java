package io.github.kirstenali.deepj.examples;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class FormatAlpacaText {

    private static final String HEADER = "Below is an instruction that describes a task. "
            + "Write a response that appropriately completes the request.";
    private static final String CONTEXT_HEADER = "Below is an instruction that describes a task, "
            + "paired with an input that provides further context. Write a response that "
            + "appropriately completes the request.";
    private static final String END_TOKEN = "<|endoftext|>";

    private FormatAlpacaText() {}

    public static void main(String[] args) throws IOException {
        Path input = Path.of(System.getProperty("deepj.alpacaInput", "sample_data/Alpaca.txt"));
        Path output = Path.of(System.getProperty(
                "deepj.alpacaOutput", "checkpoints/alpaca-deepseek/alpaca-formatted.txt"));
        FormatResult result = format(input, output);
        System.out.printf("Formatted %,d Alpaca records in %s; skipped %,d incomplete records%n",
                result.formatted(), output, result.skipped());
    }

    public static FormatResult format(Path input, Path output) throws IOException {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output");
        Path parent = output.toAbsolutePath().getParent();
        if (parent != null) Files.createDirectories(parent);
        try (BufferedReader reader = Files.newBufferedReader(input);
             BufferedWriter writer = Files.newBufferedWriter(output)) {
            return format(reader, writer);
        }
    }

    private static FormatResult format(BufferedReader reader, BufferedWriter writer)
            throws IOException {
        Parser parser = new Parser(writer);
        String line;
        while ((line = reader.readLine()) != null) parser.accept(line);
        return parser.finish();
    }

    private static boolean isHeader(String line) {
        return HEADER.equals(line) || CONTEXT_HEADER.equals(line);
    }

    private static final class Parser {

        private final BufferedWriter writer;
        private AlpacaRecord current;
        private long count;
        private long skipped;

        private Parser(BufferedWriter writer) {
            this.writer = writer;
        }

        private void accept(String line) throws IOException {
            if (isHeader(line)) {
                startRecord(CONTEXT_HEADER.equals(line));
            } else if (current != null) {
                current.add(line);
            }
        }

        private void startRecord(boolean hasContext) throws IOException {
            writeCurrent();
            current = new AlpacaRecord(hasContext);
        }

        private FormatResult finish() throws IOException {
            writeCurrent();
            writer.flush();
            return new FormatResult(count, skipped);
        }

        private void writeCurrent() throws IOException {
            if (current == null) return;
            if (current.writeTo(writer)) count++; else skipped++;
            current = null;
        }
    }

    private static final class AlpacaRecord {

        private final boolean hasContext;
        private final List<String> paragraphs = new ArrayList<>();
        private final StringBuilder paragraph = new StringBuilder();

        private AlpacaRecord(boolean hasContext) {
            this.hasContext = hasContext;
        }

        private void add(String line) {
            if (line.isBlank()) {
                finishParagraph();
            } else {
                if (!paragraph.isEmpty()) paragraph.append('\n');
                paragraph.append(line);
            }
        }

        private void finishParagraph() {
            if (paragraph.isEmpty()) return;
            paragraphs.add(paragraph.toString());
            paragraph.setLength(0);
        }

        private boolean writeTo(BufferedWriter writer) throws IOException {
            finishParagraph();
            int responseIndex = hasContext ? 2 : 1;
            if (paragraphs.size() <= responseIndex) return false;
            writeSection(writer, "Instruction", paragraphs.get(0));
            if (hasContext) writeSection(writer, "Input", paragraphs.get(1));
            writeSection(writer, "Response", responseFrom(responseIndex));
            writer.write(END_TOKEN);
            writer.newLine();
            writer.newLine();
            return true;
        }

        private String responseFrom(int start) {
            return String.join("\n\n", paragraphs.subList(start, paragraphs.size()));
        }

        private static void writeSection(BufferedWriter writer, String name, String value)
                throws IOException {
            writer.write(name);
            writer.write(":\n");
            writer.write(value);
            writer.newLine();
        }
    }

    public record FormatResult(long formatted, long skipped) {}
}
