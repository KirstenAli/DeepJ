package io.github.kirstenali.deepj.examples;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

public final class PrepareKnowledgeCorpus {

    private static final String END_TOKEN = "<|endoftext|>";
    private static final int VALIDATION_EVERY = 20;
    private static final long MEBIBYTE = 1024L * 1024L;

    private PrepareKnowledgeCorpus() {}

    public static void main(String[] args) throws IOException {
        Path output = path("deepj.output", "checkpoints/knowledge-deepseek");
        PreparationResult result = prepare(
                path("deepj.alpacaInput", "sample_data/Alpaca.txt"),
                path("deepj.storyInput", "sample_data/TinyStories-train.txt"), output,
                Integer.getInteger("deepj.storyMiB", 32),
                Integer.getInteger("deepj.factMaximum", 100),
                Integer.getInteger("deepj.factRepeats", 4));
        printResult(result, output);
    }

    public static PreparationResult prepare(Path alpaca, Path stories, Path output,
                                            int storyMiB, int factMaximum, int factRepeats)
            throws IOException {
        validateInputs(alpaca, stories, storyMiB, factRepeats);
        Files.createDirectories(output);
        Paths paths = new Paths(output);
        var formatted = FormatAlpacaText.format(alpaca, paths.formatted());
        var facts = GenerateFactDataset.generate(paths.facts(), factMaximum);
        SplitResult alpacaSplit = splitRecords(paths.formatted(), paths.alpacaTrain(),
                paths.alpacaValidation(), VALIDATION_EVERY);
        SplitResult factSplit = splitRecords(paths.facts(), paths.factTrain(),
                paths.factValidation(), VALIDATION_EVERY);
        long bytes = buildTrainingCorpus(paths, stories, storyMiB * MEBIBYTE, factRepeats);
        buildValidationCorpus(paths);
        return new PreparationResult(formatted, facts, alpacaSplit, factSplit, bytes);
    }

    private static long buildTrainingCorpus(Paths paths, Path stories, long storyBytes,
                                            int factRepeats) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(paths.training())) {
            long bytes = copyAll(paths.factTrain(), writer);
            bytes += copyAll(paths.alpacaTrain(), writer);
            for (int repeat = 1; repeat < factRepeats; repeat++) {
                bytes += copyAll(paths.factTrain(), writer);
            }
            bytes += copyRecords(stories, writer, storyBytes);
            return bytes;
        }
    }

    private static void buildValidationCorpus(Paths paths) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(paths.validation())) {
            copyAll(paths.alpacaValidation(), writer);
            copyAll(paths.factValidation(), writer);
        }
    }

    private static SplitResult splitRecords(Path input, Path training, Path validation,
                                            int validationEvery) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(input);
             BufferedWriter trainWriter = Files.newBufferedWriter(training);
             BufferedWriter validationWriter = Files.newBufferedWriter(validation)) {
            return splitRecords(reader, trainWriter, validationWriter, validationEvery);
        }
    }

    private static SplitResult splitRecords(BufferedReader reader, BufferedWriter training,
                                            BufferedWriter validation, int validationEvery)
            throws IOException {
        StringBuilder record = new StringBuilder();
        long count = 0;
        long validationCount = 0;
        String line;
        while ((line = reader.readLine()) != null) {
            if (record.isEmpty() && line.isBlank()) continue;
            record.append(line).append('\n');
            if (!END_TOKEN.equals(line)) continue;
            if (count % validationEvery == 0) validationCount++;
            writeRecord(record, count++ % validationEvery == 0 ? validation : training);
        }
        return new SplitResult(count - validationCount, validationCount);
    }

    private static void writeRecord(StringBuilder record, BufferedWriter writer)
            throws IOException {
        writer.write(record.toString());
        writer.newLine();
        record.setLength(0);
    }

    private static long copyAll(Path input, BufferedWriter writer) throws IOException {
        return copyRecords(input, writer, Long.MAX_VALUE);
    }

    private static long copyRecords(Path input, BufferedWriter writer, long minimumBytes)
            throws IOException {
        if (minimumBytes <= 0) return 0;
        long bytes = 0;
        try (BufferedReader reader = Files.newBufferedReader(input)) {
            String line;
            while ((line = reader.readLine()) != null) {
                writer.write(line);
                writer.newLine();
                bytes += line.getBytes(StandardCharsets.UTF_8).length + 1L;
                if (bytes >= minimumBytes && END_TOKEN.equals(line)) break;
            }
        }
        return bytes;
    }

    private static void validateInputs(Path alpaca, Path stories, int storyMiB, int factRepeats) {
        if (!Files.isRegularFile(alpaca)) throw new IllegalArgumentException("Alpaca file not found");
        if (!Files.isRegularFile(stories)) throw new IllegalArgumentException("TinyStories file not found");
        if (storyMiB < 1) throw new IllegalArgumentException("storyMiB must be positive");
        if (factRepeats < 1) throw new IllegalArgumentException("factRepeats must be positive");
    }

    private static void printResult(PreparationResult result, Path output) {
        System.out.printf("Alpaca: %,d training and %,d validation records%n",
                result.alpacaSplit().training(), result.alpacaSplit().validation());
        System.out.printf("Facts: %,d training and %,d validation records%n",
                result.factSplit().training(), result.factSplit().validation());
        System.out.printf("Training corpus: %.1f MiB in %s%n",
                result.corpusBytes() / (double) MEBIBYTE, output);
    }

    private static Path path(String name, String fallback) {
        return Path.of(System.getProperty(name, fallback));
    }

    public record PreparationResult(
            FormatAlpacaText.FormatResult formatted,
            GenerateFactDataset.GenerationResult facts,
            SplitResult alpacaSplit,
            SplitResult factSplit,
            long corpusBytes
    ) {}

    public record SplitResult(long training, long validation) {}

    private record Paths(Path directory) {
        private Path formatted() { return directory.resolve("alpaca-formatted.txt"); }
        private Path alpacaTrain() { return directory.resolve("alpaca-train.txt"); }
        private Path alpacaValidation() { return directory.resolve("alpaca-valid.txt"); }
        private Path facts() { return directory.resolve("facts.txt"); }
        private Path factTrain() { return directory.resolve("facts-train.txt"); }
        private Path factValidation() { return directory.resolve("facts-valid.txt"); }
        private Path training() { return directory.resolve("knowledge-train.txt"); }
        private Path validation() { return directory.resolve("knowledge-valid.txt"); }
    }
}
