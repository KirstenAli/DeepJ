package io.github.kirstenali.deepj.data;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public record TextFileRange(long startInclusive, long endExclusive) {

    public TextFileRange {
        if (startInclusive < 0 || endExclusive <= startInclusive) {
            throw new IllegalArgumentException("text range must be non-empty");
        }
    }

    public static TextFileRange entire(Path path) throws IOException {
        return new TextFileRange(0, Files.size(path));
    }

    public long length() {
        return endExclusive - startInclusive;
    }

    void validateFor(long fileSize) {
        if (endExclusive > fileSize) {
            throw new IllegalArgumentException("text range exceeds file size");
        }
    }
}
