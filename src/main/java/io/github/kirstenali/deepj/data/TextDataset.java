package io.github.kirstenali.deepj.data;

import io.github.kirstenali.deepj.tokenizers.Tokenizer;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
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
 */
public final class TextDataset {

    private static final int READ_BUFFER_CHARS  = 8_192;
    private static final int WRITE_BUFFER_BYTES = 1024 * 1024; // 1 MB → ~270K ints per flush

    /**
     * Random-access int buffer backed by one or more memory-mapped file segments.
     * Each segment is at most {@code CHUNK_BYTES} bytes, keeping individual map
     * sizes within Java's {@code Integer.MAX_VALUE} limit.
     */
    private record ChunkedIntBuffer(IntBuffer[] chunks, long intsPerChunk) {
        /** Largest multiple of 4 that fits in a signed 32-bit length: 2 147 483 644. */
        static final long CHUNK_BYTES = 0x7FFFFFFCL;

        /**
         * Core implementation. {@code chunkBytes} controls segment size; production
         * callers pass {@link #CHUNK_BYTES}, tests may pass a smaller value to
         * exercise the multi-segment path without needing a multi-gigabyte file.
         */
        static ChunkedIntBuffer map(Path file, long chunkBytes) throws IOException {
            try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
                long fileSize    = ch.size();
                long intsPerChunk = chunkBytes / Integer.BYTES;
                int  numChunks   = (int) Math.ceil((double) fileSize / chunkBytes);
                IntBuffer[] bufs = new IntBuffer[numChunks];
                for (int i = 0; i < numChunks; i++) {
                    long pos = i * chunkBytes;
                    long len = Math.min(chunkBytes, fileSize - pos);
                    bufs[i] = ch.map(FileChannel.MapMode.READ_ONLY, pos, len).asIntBuffer();
                }
                return new ChunkedIntBuffer(bufs, intsPerChunk);
            }
        }

        int get(long index) {
            int chunkIdx = (int) (index / intsPerChunk);
            int intOff   = (int) (index % intsPerChunk);
            return chunks[chunkIdx].get(intOff);
        }
    }

    private final ChunkedIntBuffer tokens;
    private final long tokenCount;
    private final int seqLen;
    private final Random rnd;

    // ── constructors ────────────────────────────────────────────────

    private TextDataset(ChunkedIntBuffer tokens, long tokenCount, int seqLen, long seed) {
        validateArgs(tokenCount, seqLen);
        this.tokens     = tokens;
        this.tokenCount = tokenCount;
        this.seqLen     = seqLen;
        this.rnd        = new Random(seed);
    }

    private static void validateArgs(long tokenCount, int seqLen) {
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
        return fromFile(path, tok, seqLen, seed, ChunkedIntBuffer.CHUNK_BYTES);
    }

    /** Package-private test hook: passes {@code chunkBytes} through to {@link ChunkedIntBuffer#map(Path, long)}. */
    static TextDataset fromFile(Path path, Tokenizer tok, int seqLen, long seed,
                                long chunkBytes) throws IOException {
        Path binFile = Files.createTempFile("deepj-tokens-", ".bin");
        binFile.toFile().deleteOnExit();
        tokenizeToFile(path, tok, binFile);
        return fromBinaryFile(binFile, seqLen, seed, chunkBytes);
    }

    /** Package-private: maps a pre-tokenized binary file directly, without re-tokenizing. */
    static TextDataset fromBinaryFile(Path binFile, int seqLen, long seed) throws IOException {
        return fromBinaryFile(binFile, seqLen, seed, ChunkedIntBuffer.CHUNK_BYTES);
    }

    /** Package-private test hook. */
    static TextDataset fromBinaryFile(Path binFile, int seqLen, long seed, long chunkBytes) throws IOException {
        long tokenCount = Files.size(binFile) / Integer.BYTES;
        return new TextDataset(ChunkedIntBuffer.map(binFile, chunkBytes), tokenCount, seqLen, seed);
    }

    // ── streaming tokenization ──────────────────────────────────────

    private static void tokenizeToFile(Path textPath, Tokenizer tok, Path binPath) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(textPath, StandardCharsets.UTF_8);
             FileChannel out = FileChannel.open(binPath, StandardOpenOption.WRITE)) {
            streamEncodeAndWrite(reader, tok, out);
        }
    }

    private static void streamEncodeAndWrite(BufferedReader reader, Tokenizer tok, FileChannel out)
            throws IOException {
        ByteBuffer writeBuf = ByteBuffer.allocate(WRITE_BUFFER_BYTES);
        StringBuilder pending = new StringBuilder();
        char[] readBuf = new char[READ_BUFFER_CHARS];
        int n;

        while ((n = reader.read(readBuf, 0, readBuf.length)) != -1) {
            pending.append(readBuf, 0, n);
            flushCompleteLines(pending, tok, writeBuf, out);
        }

        flushRemainder(pending, tok, writeBuf, out);
        flushWriteBuffer(writeBuf, out);
    }

    /** Encode and write all complete lines (up to last {@code \n}), leaving the rest in pending. */
    private static void flushCompleteLines(StringBuilder pending, Tokenizer tok,
                                           ByteBuffer writeBuf, FileChannel out) throws IOException {
        int lastNl = pending.lastIndexOf("\n");
        if (lastNl < 0) return;
        String chunk = pending.substring(0, lastNl + 1);
        pending.delete(0, lastNl + 1);
        encodeAndWrite(tok.encode(chunk), writeBuf, out);
    }

    /** Encode and write any trailing text after the last newline. */
    private static void flushRemainder(StringBuilder pending, Tokenizer tok,
                                       ByteBuffer writeBuf, FileChannel out) throws IOException {
        if (pending.isEmpty()) return;
        encodeAndWrite(tok.encode(pending.toString()), writeBuf, out);
    }

    /** Write encoded token ids to channel through a reusable buffer. */
    private static void encodeAndWrite(int[] ids, ByteBuffer buf, FileChannel ch) throws IOException {
        for (int id : ids) {
            if (buf.remaining() < Integer.BYTES) {
                flushWriteBuffer(buf, ch);
            }
            buf.putInt(id);
        }
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

        long maxStart = tokenCount - (seqLen + 1L);

        for (int b = 0; b < batchSize; b++) {
            long start = rnd.nextLong(maxStart + 1L);
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

    public long size() {
        return tokenCount;
    }
}
