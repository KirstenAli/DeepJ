package io.github.kirstenali.deepj.examples;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GenerateFactDatasetTest {

    @TempDir
    Path tempDir;

    @Test
    void generatesCheckedArithmeticAndReferenceFacts() throws Exception {
        Path output = tempDir.resolve("facts.txt");

        var result = GenerateFactDataset.generate(output, 2);
        String text = Files.readString(output);

        assertEquals(30, result.arithmetic());
        assertEquals(59, result.reference());
        assertEquals(89, result.total());
        assertTrue(text.contains("What is 2 multiplied by 2?\nResponse:\n4."));
        assertTrue(text.contains("What is the atomic number of Calcium?\nResponse:\n20."));
        assertEquals(result.total(), text.lines().filter("<|endoftext|>"::equals).count());
    }

    @Test
    void rejectsNonPositiveMaximum() {
        Path output = tempDir.resolve("facts.txt");

        assertThrows(IllegalArgumentException.class,
                () -> GenerateFactDataset.generate(output, 0));
    }

    @Test
    void rejectsUnreasonablyLargeMaximum() {
        Path output = tempDir.resolve("facts.txt");

        assertThrows(IllegalArgumentException.class,
                () -> GenerateFactDataset.generate(output, 1_001));
    }
}
