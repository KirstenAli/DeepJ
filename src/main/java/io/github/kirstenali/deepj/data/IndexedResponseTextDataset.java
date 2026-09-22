package io.github.kirstenali.deepj.data;

import io.github.kirstenali.deepj.tokenizers.Tokenizer;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public final class IndexedResponseTextDataset implements StatefulBatchSource, AutoCloseable {

    private static final String END_TOKEN = "<|endoftext|>";
    private static final String RESPONSE_MARKER = "Response:\n";
    private static final byte[] END_BYTES = END_TOKEN.getBytes(StandardCharsets.UTF_8);
    private static final int BUFFER_BYTES = 8 * 1024 * 1024;
    private static final int SAMPLE_ATTEMPTS = 128;

    private final List<Group> groups;
    private final StatefulRandom random;
    private final long totalWeight;
    private final long recordCount;

    public IndexedResponseTextDataset(List<ResponseOnlyTextDataset.Source> sources,
                                      Tokenizer tokenizer, int maxSequenceLength, long seed)
            throws IOException {
        validate(sources, tokenizer, maxSequenceLength);
        this.groups = openGroups(sources, tokenizer, maxSequenceLength);
        this.random = new StatefulRandom(seed);
        this.totalWeight = groups.stream().mapToLong(Group::weight).sum();
        this.recordCount = groups.stream().mapToLong(group -> group.file().size()).sum();
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

    public long recordCount() {
        return recordCount;
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
            if (selected < group.weight()) return group.file().sample(random);
            selected -= group.weight();
        }
        throw new IllegalStateException("Could not select a dataset group");
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

    private static List<Group> openGroups(List<ResponseOnlyTextDataset.Source> sources,
                                          Tokenizer tokenizer, int maxSequenceLength)
            throws IOException {
        List<Group> groups = new ArrayList<>(sources.size());
        for (var source : sources) {
            var file = new IndexedFile(source.path(), tokenizer, maxSequenceLength);
            if (file.size() == 0) throw new IllegalArgumentException("No records in " + source.path());
            groups.add(new Group(file, source.weight()));
        }
        return List.copyOf(groups);
    }

    private static void validate(List<ResponseOnlyTextDataset.Source> sources,
                                 Tokenizer tokenizer, int maxSequenceLength) {
        if (sources == null || sources.isEmpty()) throw new IllegalArgumentException("sources are required");
        Objects.requireNonNull(tokenizer, "tokenizer");
        if (maxSequenceLength < 2) throw new IllegalArgumentException("maxSequenceLength must be at least 2");
        for (var source : sources) {
            if (source == null || !Files.isRegularFile(source.path())) {
                throw new IllegalArgumentException("Corpus not found");
            }
            if (source.weight() < 1) throw new IllegalArgumentException("source weight must be positive");
        }
    }

    @Override
    public void close() throws IOException {
        IOException failure = null;
        for (Group group : groups) {
            try {
                group.file().close();
            } catch (IOException error) {
                failure = error;
            }
        }
        if (failure != null) throw failure;
    }

    private record Group(IndexedFile file, int weight) {}

    private record Example(int[] tokens, int responseStart) {}

    private static final class IndexedFile implements AutoCloseable {

        private final FileChannel channel;
        private final RecordIndex index;
        private final Tokenizer tokenizer;
        private final int maxSequenceLength;

        private IndexedFile(Path path, Tokenizer tokenizer, int maxSequenceLength)
                throws IOException {
            this.index = index(path);
            this.channel = FileChannel.open(path, StandardOpenOption.READ);
            this.tokenizer = tokenizer;
            this.maxSequenceLength = maxSequenceLength;
        }

        private int size() {
            return index.size();
        }

        private Example sample(StatefulRandom random) {
            for (int attempt = 0; attempt < SAMPLE_ATTEMPTS; attempt++) {
                int selected = (int) random.nextLong(index.size());
                Range range = index.range(selected);
                if (range.length() > maxSequenceLength * 64) continue;
                Example example = parse(read(range), tokenizer);
                if (example != null && example.tokens().length <= maxSequenceLength + 1) return example;
            }
            throw new IllegalStateException("Could not sample a usable response record");
        }

        private String read(Range range) {
            try {
                ByteBuffer buffer = ByteBuffer.allocate(range.length());
                readFully(buffer, range.start());
                return new String(buffer.array(), StandardCharsets.UTF_8).strip();
            } catch (IOException error) {
                throw new IllegalStateException("Could not read response record", error);
            }
        }

        private void readFully(ByteBuffer buffer, long position) throws IOException {
            while (buffer.hasRemaining()) {
                int read = channel.read(buffer, position + buffer.position());
                if (read <= 0) throw new IOException("Unexpected end of response corpus");
            }
        }

        @Override
        public void close() throws IOException {
            channel.close();
        }
    }

    private static Example parse(String record, Tokenizer tokenizer) {
        int marker = record.indexOf(RESPONSE_MARKER);
        if (marker < 0) return null;
        int responseStart = marker + RESPONSE_MARKER.length();
        String prompt = record.substring(0, responseStart);
        String response = record.substring(responseStart).strip();
        if (response.isEmpty()) return null;
        return encode(prompt, response, tokenizer);
    }

    private static Example encode(String prompt, String response, Tokenizer tokenizer) {
        int[] promptTokens = tokenizer.encode(prompt);
        int[] responseTokens = tokenizer.encode(response + "\n" + END_TOKEN);
        int[] tokens = Arrays.copyOf(promptTokens, promptTokens.length + responseTokens.length);
        System.arraycopy(responseTokens, 0, tokens, promptTokens.length, responseTokens.length);
        return new Example(tokens, promptTokens.length);
    }

    private static RecordIndex index(Path path) throws IOException {
        IndexBuilder builder = new IndexBuilder();
        ScanState state = new ScanState();
        try (InputStream input = new BufferedInputStream(Files.newInputStream(path), BUFFER_BYTES)) {
            byte[] buffer = new byte[BUFFER_BYTES];
            for (int read = input.read(buffer); read >= 0; read = input.read(buffer)) {
                scan(buffer, read, state, builder);
            }
        }
        return builder.build();
    }

    private static void scan(byte[] buffer, int length, ScanState state, IndexBuilder builder) {
        for (int offset = 0; offset < length; offset++) {
            byte value = buffer[offset];
            state.matched = value == END_BYTES[state.matched] ? state.matched + 1
                    : value == END_BYTES[0] ? 1 : 0;
            if (state.matched == END_BYTES.length) addRange(state, builder);
            state.position++;
        }
    }

    private static void addRange(ScanState state, IndexBuilder builder) {
        long markerStart = state.position - END_BYTES.length + 1;
        long length = markerStart - state.recordStart;
        if (length > Integer.MAX_VALUE) throw new IllegalArgumentException("Response record is too large");
        builder.add(state.recordStart, (int) length);
        state.recordStart = state.position + 1;
        state.matched = 0;
    }

    private record Range(long start, int length) {}

    private record RecordIndex(long[] starts, int[] lengths) {

        private int size() {
            return starts.length;
        }

        private Range range(int index) {
            return new Range(starts[index], lengths[index]);
        }
    }

    private static final class IndexBuilder {

        private long[] starts = new long[1_024];
        private int[] lengths = new int[1_024];
        private int size;

        private void add(long start, int length) {
            ensureCapacity();
            starts[size] = start;
            lengths[size] = length;
            size++;
        }

        private void ensureCapacity() {
            if (size < starts.length) return;
            int capacity = Math.multiplyExact(size, 2);
            starts = Arrays.copyOf(starts, capacity);
            lengths = Arrays.copyOf(lengths, capacity);
        }

        private RecordIndex build() {
            return new RecordIndex(Arrays.copyOf(starts, size), Arrays.copyOf(lengths, size));
        }
    }

    private static final class ScanState {

        private long position;
        private long recordStart;
        private int matched;
    }
}
