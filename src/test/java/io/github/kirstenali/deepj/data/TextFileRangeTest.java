package io.github.kirstenali.deepj.data;

import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class TextFileRangeTest {

    @Test
    void entireRangeUsesFileSize() throws Exception {
        Path path = Files.createTempFile("deepj-range", ".txt");
        Files.writeString(path, "validation text");
        assertEquals(new TextFileRange(0, 15), TextFileRange.entire(path));
    }

    @Test
    void rejectsEmptyAndOutOfBoundsRanges() {
        assertThrows(IllegalArgumentException.class, () -> new TextFileRange(2, 2));
        var range = new TextFileRange(2, 5);
        assertThrows(IllegalArgumentException.class, () -> range.validateFor(4));
    }
}
