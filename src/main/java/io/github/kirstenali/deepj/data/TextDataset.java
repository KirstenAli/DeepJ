package io.github.kirstenali.deepj.data;

import io.github.kirstenali.deepj.tokenizers.Tokenizer;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Random;

/**
 * Dataset that samples random contiguous chunks from token ids.
 *
 * <p>When created via {@link #fromFile}, the source text is streamed line-by-line,
 * tokenized in bounded chunks, and written to a temporary binary file that is then
 * memory-mapped. The full file never needs to fit in Java heap — the OS pages the
 * token data in and out as needed.
 *
 * <p>The {@code int[]} constructor wraps the array directly for small/test data.
 */
public final class TextDataset {

    private static final int IO_BUFFER_SIZE = 8192;

    private final IntBuffer tokens;
    private final int tokenCount;
    private final int seqLen;
    private final Random rnd;

    // ── constructors ────────────────────────────────────────────────

    public TextDataset(int[] tokens, int seqLen, long seed) {
        validateArgs(tokens.length, seqLen);
        this.tokens = IntBuffer.wrap(tokens);
        this.tokenCount = tokens.length;
        this.seqLen = seqLen;
        this.rnd = new Random(seed);
    }

    private TextDataset(IntBuffer tokens, int tokenCount, int seqLen, long seed) {
        this.tokens = tokens;
        this.tokenCount = tokenCount;
        this.seqLen = seqLen;
        this.rnd = new Random(seed);
    }

    private static void validateArgs(int tokenCount, int seqLen) {
        if (seqLen < 2) throw new IllegalArgumentException("seqLen must be >= 2");
        if (tokenCount < seqLen + 1) throw new IllegalArgumentException("Not enough tokens for seqLen+1");
    }

    // ── factory ─────────────────────────────────────────────────────

    /**
     * Stream-tokenize a text file and memory-map the result.
     * The text is read in bounded chunks (split on line boundaries) so that
     * neither the raw text nor the full token array need to fit in heap.
     */
    public static TextDataset fromFile(Path path, Tokenizer tok, int seqLen, long seed) throws IOException {
        Path binFile = createTempTokenFile();
        int tokenCount = tokenizeToFile(path, tok, binFile);
        validateArgs(tokenCount, seqLen);
        IntBuffer mapped = memoryMap(binFile);
        return new TextDataset(mapped, tokenCount, seqLen, seed);
    }

    private static Path createTempTokenFile() throws IOException {
        Path p = Files.createTempFile("deepj-tokens-", ".bin");
        p.toFile().deleteOnExit();
        return p;
    }

    private static IntBuffer memoryMap(Path binFile) throws IOException {
        try (FileChannel ch = FileChannel.open(binFile, StandardOpenOption.READ)) {
            MappedByteBuffer mapped = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size());
            return mapped.asIntBuffer();
        }
    }

    // ── streaming tokenization ──────────────────────────────────────

    private static int tokenizeToFile(Path textPath, Tokenizer tok, Path binPath) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(textPath, StandardCharsets.UTF_8);
             FileChannel out = FileChannel.open(binPath, StandardOpenOption.WRITE)) {
            return streamEncodeAndWrite(reader, tok, out);
        }
    }

    private static int streamEncodeAndWrite(BufferedReader reader, Tokenizer tok, FileChannel out)
            throws IOException {
        ByteBuffer writeBuf = ByteBuffer.allocate(IO_BUFFER_SIZE);
        StringBuilder pending = new StringBuilder();
        char[] readBuf = new char[IO_BUFFER_SIZE];
        int totalTokens = 0;
        int n;

        while ((n = reader.read(readBuf, 0, readBuf.length)) != -1) {
            pending.append(readBuf, 0, n);
            totalTokens += flushCompleteLines(pending, tok, writeBuf, out);
        }

        totalTokens += flushRemainder(pending, tok, writeBuf, out);
        flushWriteBuffer(writeBuf, out);
        return totalTokens;
    }

    /** Encode and write all complete lines (up to last {@code \n}), leaving the rest in pending. */
    private static int flushCompleteLines(StringBuilder pending, Tokenizer tok,
                                          ByteBuffer writeBuf, FileChannel out) throws IOException {
        int lastNl = pending.lastIndexOf("\n");
        if (lastNl < 0) return 0;

        String chunk = pending.substring(0, lastNl + 1);
        pending.delete(0, lastNl + 1);
        return encodeAndWrite(tok.encode(chunk), writeBuf, out);
    }

    /** Encode and write any trailing text after the last newline. */
    private static int flushRemainder(StringBuilder pending, Tokenizer tok,
                                      ByteBuffer writeBuf, FileChannel out) throws IOException {
        if (pending.isEmpty()) return 0;
        return encodeAndWrite(tok.encode(pending.toString()), writeBuf, out);
    }

    /** Write encoded token ids to channel through a reusable buffer. */
    private static int encodeAndWrite(int[] ids, ByteBuffer buf, FileChannel ch) throws IOException {
        for (int id : ids) {
            if (buf.remaining() < Integer.BYTES) {
                flushWriteBuffer(buf, ch);
            }
            buf.putInt(id);
        }
        return ids.length;
    }

    private static void flushWriteBuffer(ByteBuffer buf, FileChannel ch) throws IOException {
        buf.flip();
        while (buf.hasRemaining()) ch.write(buf);
        buf.clear();
    }

    // ── batch sampling ──────────────────────────────────────────────

    public Batch nextBatch(int batchSize) {
        int[][] x = new int[batchSize][seqLen];
        int[][] y = new int[batchSize][seqLen];

        int maxStart = tokenCount - (seqLen + 1);

        for (int b = 0; b < batchSize; b++) {
            int start = rnd.nextInt(maxStart + 1);
            for (int t = 0; t < seqLen; t++) {
                x[b][t] = tokens.get(start + t);
                y[b][t] = tokens.get(start + t + 1);
            }
        }
        return new Batch(x, y);
    }

    public int seqLen() {
        return seqLen;
    }

    public int size() {
        return tokenCount;
    }
}
