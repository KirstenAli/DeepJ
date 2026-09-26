package io.github.kirstenali.deepj.data;

import io.github.kirstenali.deepj.tokenizers.Tokenizer;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Objects;

public final class RandomAccessTextDataset implements StatefulTrainingDataset {

    private static final int MIN_WINDOW_BYTES = 4 * 1024;
    private static final int MAX_WINDOW_BYTES = 8 * 1024 * 1024;
    private static final int MAX_ATTEMPTS = 12;

    private final Tokenizer tokenizer;
    private final int seqLen;
    private final StatefulRandom random;
    private final FileChannel channel;
    private final long rangeStart;
    private final long rangeEnd;

    public RandomAccessTextDataset(Path path, Tokenizer tokenizer, int seqLen, long seed)
            throws IOException {
        this(path, tokenizer, seqLen, seed, TextFileRange.entire(path));
    }

    public RandomAccessTextDataset(Path path, Tokenizer tokenizer, int seqLen, long seed,
                                   TextFileRange range) throws IOException {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        if (seqLen < 2) throw new IllegalArgumentException("seqLen must be >= 2");
        long fileSize = Files.size(Objects.requireNonNull(path, "path"));
        Objects.requireNonNull(range, "range").validateFor(fileSize);
        this.rangeStart = range.startInclusive();
        this.rangeEnd = range.endExclusive();
        this.seqLen = seqLen;
        this.random = new StatefulRandom(seed);
        this.channel = FileChannel.open(path, StandardOpenOption.READ);
    }

    @Override
    public synchronized Batch nextBatch(int batchSize) {
        if (batchSize < 1) throw new IllegalArgumentException("batchSize must be >= 1");
        try {
            return batchFrom(sampleTokens(requiredTokens(batchSize)), batchSize);
        } catch (IOException e) {
            throw new UncheckedIOException("Could not sample training text", e);
        }
    }

    private int[] sampleTokens(int required) throws IOException {
        int window = initialWindow(required);
        for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
            int[] tokens = tokenizer.encode(readWindow(window));
            if (tokens.length >= required) return tokens;
            window = Math.min(MAX_WINDOW_BYTES, window * 2);
        }
        throw new IOException("Could not read enough tokens for one batch");
    }

    private String readWindow(int requestedBytes) throws IOException {
        int length = (int) Math.min(rangeEnd - rangeStart, requestedBytes);
        long position = randomPosition(length);
        byte[] bytes = readBytes(position, length);
        int start = position == rangeStart ? 0 : indexAfterFirstNewline(bytes);
        int end = position + bytes.length == rangeEnd ? bytes.length : indexAfterLastNewline(bytes);
        if (start < 0 || end <= start) return "";
        return new String(bytes, start, end - start, StandardCharsets.UTF_8);
    }

    private byte[] readBytes(long position, int length) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(length);
        while (buffer.hasRemaining()) {
            int read = channel.read(buffer, position + buffer.position());
            if (read <= 0) break;
        }
        return Arrays.copyOf(buffer.array(), buffer.position());
    }

    private long randomPosition(int windowLength) {
        long bound = rangeEnd - rangeStart - windowLength + 1;
        return bound <= 1 ? rangeStart : rangeStart + random.nextLong(bound);
    }

    public synchronized long randomState() {
        return random.state();
    }

    public synchronized void restoreRandomState(long state) {
        random.restore(state);
    }

    private int initialWindow(int requiredTokens) {
        long estimate = Math.max(MIN_WINDOW_BYTES, requiredTokens * 8L);
        return (int) Math.min(MAX_WINDOW_BYTES, estimate);
    }

    private int requiredTokens(int batchSize) {
        return Math.addExact(Math.multiplyExact(batchSize, seqLen), 1);
    }

    private Batch batchFrom(int[] tokens, int batchSize) {
        int[][] inputs = new int[batchSize][seqLen];
        int[][] targets = new int[batchSize][seqLen];
        for (int row = 0; row < batchSize; row++) {
            int offset = row * seqLen;
            System.arraycopy(tokens, offset, inputs[row], 0, seqLen);
            System.arraycopy(tokens, offset + 1, targets[row], 0, seqLen);
        }
        return new Batch(inputs, targets);
    }

    private static int indexAfterFirstNewline(byte[] bytes) {
        for (int i = 0; i < bytes.length; i++) {
            if (bytes[i] == '\n') return i + 1;
        }
        return -1;
    }

    private static int indexAfterLastNewline(byte[] bytes) {
        for (int i = bytes.length - 1; i >= 0; i--) {
            if (bytes[i] == '\n') return i + 1;
        }
        return -1;
    }

    @Override
    public void close() throws IOException {
        channel.close();
    }
}
