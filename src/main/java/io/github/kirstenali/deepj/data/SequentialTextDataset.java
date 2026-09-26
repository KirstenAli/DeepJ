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
import java.util.Objects;

public final class SequentialTextDataset implements StatefulTrainingDataset {

    private static final int DEFAULT_BLOCK_BYTES = 8 * 1024 * 1024;
    private static final int CURSOR_BITS = 24;
    private static final long CURSOR_MASK = (1L << CURSOR_BITS) - 1;

    private final Tokenizer tokenizer;
    private final int sequenceLength;
    private final int blockBytes;
    private final FileChannel channel;
    private final long rangeStart;
    private final long rangeEnd;

    private long blockStart;
    private long nextBlockStart;
    private int[] tokens;
    private int cursor;

    public SequentialTextDataset(Path path, Tokenizer tokenizer, int sequenceLength)
            throws IOException {
        this(path, tokenizer, sequenceLength, DEFAULT_BLOCK_BYTES, TextFileRange.entire(path));
    }

    public SequentialTextDataset(Path path, Tokenizer tokenizer, int sequenceLength,
                                 TextFileRange range) throws IOException {
        this(path, tokenizer, sequenceLength, DEFAULT_BLOCK_BYTES, range);
    }

    SequentialTextDataset(Path path, Tokenizer tokenizer, int sequenceLength,
                          int blockBytes) throws IOException {
        this(path, tokenizer, sequenceLength, blockBytes, TextFileRange.entire(path));
    }

    private SequentialTextDataset(Path path, Tokenizer tokenizer, int sequenceLength,
                                  int blockBytes, TextFileRange range) throws IOException {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        if (sequenceLength < 2) throw new IllegalArgumentException("sequenceLength must be at least 2");
        if (blockBytes < 1_024 || blockBytes > CURSOR_MASK) {
            throw new IllegalArgumentException("blockBytes is out of range");
        }
        long fileSize = Files.size(Objects.requireNonNull(path, "path"));
        Objects.requireNonNull(range, "range").validateFor(fileSize);
        this.rangeStart = range.startInclusive();
        this.rangeEnd = range.endExclusive();
        this.sequenceLength = sequenceLength;
        this.blockBytes = blockBytes;
        this.channel = FileChannel.open(path, StandardOpenOption.READ);
        loadBlock(rangeStart);
    }

    @Override
    public synchronized Batch nextBatch(int batchSize) {
        if (batchSize < 1) throw new IllegalArgumentException("batchSize must be positive");
        int required = Math.addExact(Math.multiplyExact(batchSize, sequenceLength), 1);
        int[] sampled = nextTokens(required);
        return batch(sampled, batchSize);
    }

    private int[] nextTokens(int required) {
        int[] sampled = new int[required];
        int written = 0;
        while (written < required) {
            if (cursor == tokens.length) loadNextBlock();
            int count = Math.min(required - written, tokens.length - cursor);
            System.arraycopy(tokens, cursor, sampled, written, count);
            cursor += count;
            written += count;
        }
        cursor--;
        return sampled;
    }

    private Batch batch(int[] sampled, int batchSize) {
        int[][] inputs = new int[batchSize][sequenceLength];
        int[][] targets = new int[batchSize][sequenceLength];
        for (int row = 0; row < batchSize; row++) {
            int offset = row * sequenceLength;
            System.arraycopy(sampled, offset, inputs[row], 0, sequenceLength);
            System.arraycopy(sampled, offset + 1, targets[row], 0, sequenceLength);
        }
        return new Batch(inputs, targets);
    }

    private void loadNextBlock() {
        try {
            loadBlock(nextBlockStart == rangeEnd ? rangeStart : nextBlockStart);
        } catch (IOException error) {
            throw new UncheckedIOException("Could not read sequential training text", error);
        }
    }

    private void loadBlock(long position) throws IOException {
        byte[] bytes = readBytes(position);
        int length = position + bytes.length == rangeEnd ? bytes.length : indexAfterLastNewline(bytes);
        if (length <= 0) throw new IOException("Training block contains no complete line");
        this.blockStart = position;
        this.nextBlockStart = position + length;
        this.tokens = tokenizer.encode(new String(bytes, 0, length, StandardCharsets.UTF_8));
        this.cursor = 0;
        if (tokens.length > CURSOR_MASK) throw new IOException("Training block contains too many tokens");
    }

    private byte[] readBytes(long position) throws IOException {
        int length = (int) Math.min(blockBytes, rangeEnd - position);
        ByteBuffer buffer = ByteBuffer.allocate(length);
        while (buffer.hasRemaining()) {
            int read = channel.read(buffer, position + buffer.position());
            if (read <= 0) throw new IOException("Unexpected end of training corpus");
        }
        return buffer.array();
    }

    private static int indexAfterLastNewline(byte[] bytes) {
        for (int index = bytes.length - 1; index >= 0; index--) {
            if (bytes[index] == '\n') return index + 1;
        }
        return -1;
    }

    @Override
    public synchronized long randomState() {
        return Math.addExact(Math.multiplyExact(blockStart, 1L << CURSOR_BITS), cursor);
    }

    @Override
    public synchronized void restoreRandomState(long state) {
        if (state < 0) throw new IllegalArgumentException("state must be non-negative");
        long restoredBlock = state >>> CURSOR_BITS;
        int restoredCursor = (int) (state & CURSOR_MASK);
        if (restoredBlock < rangeStart || restoredBlock >= rangeEnd) {
            throw new IllegalArgumentException("invalid dataset block");
        }
        try {
            loadBlock(restoredBlock);
        } catch (IOException error) {
            throw new UncheckedIOException("Could not restore sequential training data", error);
        }
        if (restoredCursor > tokens.length) throw new IllegalArgumentException("invalid dataset cursor");
        cursor = restoredCursor;
    }

    @Override
    public void close() throws IOException {
        channel.close();
    }
}
