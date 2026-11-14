package io.github.kirstenali.deepj.persistence.gguf;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

/** Bounded little-endian reads for GGUF v3 metadata. */
final class GgufInput {
    private static final long MAX_ITEMS = 16_777_216;
    final RandomAccessFile file;

    GgufInput(RandomAccessFile file) { this.file = file; }

    int i32() throws IOException { return Integer.reverseBytes(file.readInt()); }
    long i64() throws IOException { return Long.reverseBytes(file.readLong()); }

    int count(long limit) throws IOException {
        long value = i64();
        if (value < 0 || value > limit) throw new IOException("Invalid GGUF count: " + value);
        return (int) value;
    }

    String string() throws IOException {
        byte[] bytes = new byte[count(Math.min(MAX_ITEMS, file.length() - file.getFilePointer()))];
        file.readFully(bytes);
        return StandardCharsets.UTF_8.newDecoder().decode(ByteBuffer.wrap(bytes)).toString();
    }

    void skip(long bytes) throws IOException {
        if (bytes < 0 || bytes > file.length() - file.getFilePointer()) {
            throw new IOException("Truncated GGUF metadata");
        }
        file.seek(file.getFilePointer() + bytes);
    }

    private static int width(int type) throws IOException {
        return switch (type) {
            case 0, 1, 7 -> 1;
            case 2, 3 -> 2;
            case 4, 5, 6 -> 4;
            case 10, 11, 12 -> 8;
            default -> throw new IOException("Unsupported GGUF metadata type: " + type);
        };
    }

    Object metadata(int type) throws IOException {
        if (type == 4) return Integer.toUnsignedLong(i32());
        if (type == 6) return Float.intBitsToFloat(i32());
        if (type == 8) return string();
        if (type == 9) skipArray();
        else skip(width(type));
        return null;
    }

    private void skipArray() throws IOException {
        int type = i32(), count = count(MAX_ITEMS);
        if (type == 8) {
            for (int index = 0; index < count; index++) {
                skip(count(MAX_ITEMS));
            }
        } else {
            skip((long) count * width(type));
        }
    }
}
