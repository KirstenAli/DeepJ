package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.data.TextFileRange;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.integer;

final class DeepJ90MDataSplit {

    private static final int BUFFER_BYTES = 8_192;
    private static final int DEFAULT_VALIDATION_MIB = 64;

    private DeepJ90MDataSplit() {}

    static Split fromSystemProperties(Path corpus) throws IOException {
        int mebibytes = integer("deepj.validationMiB", DEFAULT_VALIDATION_MIB);
        return split(corpus, Math.multiplyExact((long) mebibytes, 1_024L * 1_024L));
    }

    static Split split(Path corpus, long validationBytes) throws IOException {
        long fileSize = Files.size(corpus);
        validate(fileSize, validationBytes);
        long boundary = boundary(corpus, fileSize - validationBytes, fileSize);
        return new Split(new TextFileRange(0, boundary),
                new TextFileRange(boundary, fileSize));
    }

    private static void validate(long fileSize, long validationBytes) {
        if (validationBytes <= 0 || validationBytes >= fileSize) {
            throw new IllegalArgumentException("validation size must be smaller than the corpus");
        }
    }

    private static long boundary(Path path, long position, long fileSize) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            return boundary(channel, position, fileSize);
        }
    }

    private static long boundary(FileChannel channel, long position, long fileSize)
            throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(BUFFER_BYTES);
        while (position < fileSize) {
            int read = read(channel, buffer, position, fileSize);
            int newline = newline(buffer, read);
            if (newline >= 0) return position + newline + 1;
            position += read;
        }
        throw new IOException("Could not find a validation boundary");
    }

    private static int read(FileChannel channel, ByteBuffer buffer,
                            long position, long fileSize) throws IOException {
        buffer.clear();
        buffer.limit((int) Math.min(buffer.capacity(), fileSize - position));
        int read = channel.read(buffer, position);
        if (read <= 0) throw new IOException("Could not read the validation boundary");
        return read;
    }

    private static int newline(ByteBuffer buffer, int length) {
        for (int index = 0; index < length; index++) {
            if (buffer.get(index) == '\n') return index;
        }
        return -1;
    }

    record Split(TextFileRange training, TextFileRange validation) {}
}
