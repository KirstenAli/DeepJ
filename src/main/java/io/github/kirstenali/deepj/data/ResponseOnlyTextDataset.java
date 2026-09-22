package io.github.kirstenali.deepj.data;

import io.github.kirstenali.deepj.tokenizers.Tokenizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public final class ResponseOnlyTextDataset implements StatefulBatchSource {

    private static final String END_TOKEN = "<|endoftext|>";
    private static final String RESPONSE_MARKER = "Response:\n";

    private final List<Group> groups;
    private final List<Example> examples;
    private final StatefulRandom random;
    private final long totalWeight;

    public ResponseOnlyTextDataset(List<Source> sources, Tokenizer tokenizer,
                                   int maxSequenceLength, long seed) throws IOException {
        validateArguments(sources, tokenizer, maxSequenceLength);
        this.groups = loadGroups(sources, tokenizer, maxSequenceLength);
        this.examples = groups.stream().flatMap(group -> group.examples().stream()).toList();
        this.totalWeight = groups.stream().mapToLong(Group::weight).sum();
        this.random = new StatefulRandom(seed);
    }

    @Override
    public synchronized Batch nextBatch(int batchSize) {
        if (batchSize < 1) throw new IllegalArgumentException("batchSize must be positive");
        int[][] inputs = new int[batchSize][];
        int[][] targets = new int[batchSize][];
        boolean[][] masks = new boolean[batchSize][];
        for (int row = 0; row < batchSize; row++) {
            fillRow(sample(), inputs, targets, masks, row);
        }
        return new Batch(inputs, targets, masks);
    }

    public List<Example> examples() {
        return examples;
    }

    @Override
    public synchronized long randomState() {
        return random.state();
    }

    @Override
    public synchronized void restoreRandomState(long state) {
        random.restore(state);
    }

    private Example sample() {
        long selected = random.nextLong(totalWeight);
        for (Group group : groups) {
            if (selected < group.weight()) return randomExample(group.examples());
            selected -= group.weight();
        }
        throw new IllegalStateException("Could not select a dataset group");
    }

    private Example randomExample(List<Example> choices) {
        return choices.get((int) random.nextLong(choices.size()));
    }

    private static void fillRow(Example example, int[][] inputs, int[][] targets,
                                boolean[][] masks, int row) {
        int length = example.tokens().length - 1;
        inputs[row] = Arrays.copyOf(example.tokens(), length);
        targets[row] = Arrays.copyOfRange(example.tokens(), 1, length + 1);
        masks[row] = responseMask(length, example.responseStart());
    }

    private static boolean[] responseMask(int length, int responseStart) {
        boolean[] mask = new boolean[length];
        Arrays.fill(mask, responseStart - 1, length, true);
        return mask;
    }

    private static List<Group> loadGroups(List<Source> sources, Tokenizer tokenizer,
                                          int maxSequenceLength) throws IOException {
        List<Group> groups = new ArrayList<>(sources.size());
        for (Source source : sources) {
            List<Example> loaded = load(source.path(), tokenizer, maxSequenceLength);
            if (loaded.isEmpty()) throw new IllegalArgumentException("No usable records in " + source.path());
            groups.add(new Group(loaded, source.weight()));
        }
        return List.copyOf(groups);
    }

    private static List<Example> load(Path path, Tokenizer tokenizer,
                                      int maxSequenceLength) throws IOException {
        String text = Files.readString(path);
        List<Example> loaded = new ArrayList<>();
        int start = 0;
        for (int end = text.indexOf(END_TOKEN); end >= 0; end = text.indexOf(END_TOKEN, start)) {
            addRecord(loaded, text.substring(start, end), tokenizer, maxSequenceLength);
            start = end + END_TOKEN.length();
        }
        return List.copyOf(loaded);
    }

    private static void addRecord(List<Example> target, String raw, Tokenizer tokenizer,
                                  int maxSequenceLength) {
        String record = raw.strip();
        int marker = record.indexOf(RESPONSE_MARKER);
        if (marker < 0) return;
        String prompt = record.substring(0, marker + RESPONSE_MARKER.length());
        String response = record.substring(marker + RESPONSE_MARKER.length()).strip();
        Example example = encode(prompt, response, tokenizer);
        if (!response.isEmpty() && example.tokens().length <= maxSequenceLength + 1) target.add(example);
    }

    private static Example encode(String prompt, String response, Tokenizer tokenizer) {
        int[] promptTokens = tokenizer.encode(prompt);
        int[] responseTokens = tokenizer.encode(response + "\n" + END_TOKEN);
        int[] tokens = Arrays.copyOf(promptTokens, promptTokens.length + responseTokens.length);
        System.arraycopy(responseTokens, 0, tokens, promptTokens.length, responseTokens.length);
        return new Example(prompt, response, tokens, promptTokens.length);
    }

    private static void validateArguments(List<Source> sources, Tokenizer tokenizer,
                                          int maxSequenceLength) {
        if (sources == null || sources.isEmpty()) throw new IllegalArgumentException("sources are required");
        Objects.requireNonNull(tokenizer, "tokenizer");
        if (maxSequenceLength < 2) throw new IllegalArgumentException("maxSequenceLength must be at least 2");
        for (Source source : sources) {
            validateSource(source);
        }
    }

    private static void validateSource(Source source) {
        Objects.requireNonNull(source, "source");
        if (!Files.isRegularFile(source.path())) {
            throw new IllegalArgumentException("Corpus not found: " + source.path());
        }
        if (source.weight() < 1) throw new IllegalArgumentException("source weight must be positive");
    }

    public record Source(Path path, int weight) {

        public Source {
            Objects.requireNonNull(path, "path");
        }
    }

    public record Example(String prompt, String response, int[] tokens, int responseStart) {}

    private record Group(List<Example> examples, int weight) {}
}
