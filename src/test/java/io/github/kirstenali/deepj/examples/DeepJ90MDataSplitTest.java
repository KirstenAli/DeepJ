package io.github.kirstenali.deepj.examples;

import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DeepJ90MDataSplitTest {

    @Test
    void splitReservesWholeLinesFromCorpusTail() throws Exception {
        String text = "first record\nsecond record\nthird record\n";
        Path path = Files.createTempFile("deepj-pretraining", ".txt");
        Files.writeString(path, text);
        var split = DeepJ90MDataSplit.split(path, 20);
        assertEquals(split.training().endExclusive(), split.validation().startInclusive());
        assertEquals(text.length(), split.validation().endExclusive());
        assertTrue(validationText(text, split).startsWith("third record"));
    }

    private static String validationText(String text, DeepJ90MDataSplit.Split split) {
        return text.substring((int) split.validation().startInclusive());
    }
}
