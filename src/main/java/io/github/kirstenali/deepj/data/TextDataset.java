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

public final class TextDataset implements BatchSource {

    private static final int READ_BUFFER_CHARS  = 8_192;
    private static final int WRITE_BUFFER_BYTES = 1024 * 1024;

    private record ChunkedIntBuffer(IntBuffer[] chunks, long intsPerChunk) {

        static final long CHUNK_BYTES = 0x7FFFFFFCL;

        static ChunkedIntBuffer map(Path file, long chunkBytes) throws IOException {
            validateChunkBytes(chunkBytes);
            try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ)) {
                long fileSize = channel.size();
                long intsPerChunk = chunkBytes / Integer.BYTES;
                return new ChunkedIntBuffer(mapChunks(channel, fileSize, chunkBytes), intsPerChunk);
            }
        }

        private static IntBuffer[] mapChunks(FileChannel channel, long fileSize,
                                             long chunkBytes) throws IOException {
            IntBuffer[] chunks = new IntBuffer[chunkCount(fileSize, chunkBytes)];
            for (int index = 0; index < chunks.length; index++) {
                chunks[index] = mapChunk(channel, fileSize, chunkBytes, index);
            }
            return chunks;
        }

        private static IntBuffer mapChunk(FileChannel channel, long fileSize,
                                          long chunkBytes, int index) throws IOException {
            long position = index * chunkBytes;
            long length = Math.min(chunkBytes, fileSize - position);
            return channel.map(FileChannel.MapMode.READ_ONLY, position, length).asIntBuffer();
        }

        private static int chunkCount(long fileSize, long chunkBytes) throws IOException {
            long count = divideRoundingUp(fileSize, chunkBytes);
            if (count > Integer.MAX_VALUE) {
                throw new IOException("Token file requires too many mapped segments");
            }
            return (int) count;
        }

        private static void validateChunkBytes(long chunkBytes) {
            if (chunkBytes <= 0 || chunkBytes > CHUNK_BYTES || chunkBytes % Integer.BYTES != 0) {
                throw new IllegalArgumentException("chunkBytes must be a positive multiple of 4 up to " + CHUNK_BYTES);
            }
        }

        private static long divideRoundingUp(long value, long divisor) {
            return value / divisor + (value % divisor == 0 ? 0 : 1);
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

    public static TextDataset fromFile(Path path, Tokenizer tok, int seqLen, long seed) throws IOException {
        return fromFile(path, tok, seqLen, seed, ChunkedIntBuffer.CHUNK_BYTES);
    }

    static TextDataset fromFile(Path path, Tokenizer tok, int seqLen, long seed,
                                long chunkBytes) throws IOException {
        Path binFile = Files.createTempFile("deepj-tokens-", ".bin");
        binFile.toFile().deleteOnExit();
        tokenizeToFile(path, tok, binFile);
        return fromBinaryFile(binFile, seqLen, seed, chunkBytes);
    }

    static TextDataset fromBinaryFile(Path binFile, int seqLen, long seed) throws IOException {
        return fromBinaryFile(binFile, seqLen, seed, ChunkedIntBuffer.CHUNK_BYTES);
    }

    static TextDataset fromBinaryFile(Path binFile, int seqLen, long seed, long chunkBytes) throws IOException {
        long byteCount = Files.size(binFile);
        if (byteCount % Integer.BYTES != 0) {
            throw new IOException("Malformed token file: byte length must be divisible by 4");
        }
        long tokenCount = byteCount / Integer.BYTES;
        return new TextDataset(ChunkedIntBuffer.map(binFile, chunkBytes), tokenCount, seqLen, seed);
    }

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

    private static void flushCompleteLines(StringBuilder pending, Tokenizer tok,
                                           ByteBuffer writeBuf, FileChannel out) throws IOException {
        int lastNl = pending.lastIndexOf("\n");
        if (lastNl < 0) return;
        String chunk = pending.substring(0, lastNl + 1);
        pending.delete(0, lastNl + 1);
        encodeAndWrite(tok.encode(chunk), writeBuf, out);
    }

    private static void flushRemainder(StringBuilder pending, Tokenizer tok,
                                       ByteBuffer writeBuf, FileChannel out) throws IOException {
        if (pending.isEmpty()) return;
        encodeAndWrite(tok.encode(pending.toString()), writeBuf, out);
    }

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
        while (buf.hasRemaining()) {
            ch.write(buf);
        }
        buf.clear();
    }

    public Batch nextBatch(int batchSize) {
        if (batchSize < 1) throw new IllegalArgumentException("batchSize must be >= 1");
        int[][] inputs = new int[batchSize][seqLen];
        int[][] targets = new int[batchSize][seqLen];
        long maxStart = tokenCount - (seqLen + 1L);
        for (int row = 0; row < batchSize; row++) {
            sampleRow(inputs[row], targets[row], maxStart);
        }
        return new Batch(inputs, targets);
    }

    private void sampleRow(int[] inputs, int[] targets, long maxStart) {
        long start = rnd.nextLong(maxStart + 1L);
        for (int token = 0; token < seqLen; token++) {
            inputs[token] = tokens.get(start + token);
            targets[token] = tokens.get(start + token + 1);
        }
    }

    public int seqLen() {
        return seqLen;
    }

    public long size() {
        return tokenCount;
    }
}
