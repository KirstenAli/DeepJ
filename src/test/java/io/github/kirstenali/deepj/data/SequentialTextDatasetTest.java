package io.github.kirstenali.deepj.data;

import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class SequentialTextDatasetTest {

    @TempDir
    Path temporaryDirectory;

    @Test
    void consecutiveBatchesShareBoundaryToken() throws Exception {
        try (SequentialTextDataset dataset = dataset(4)) {
            Batch first = dataset.nextBatch(1);
            Batch second = dataset.nextBatch(1);
            assertEquals(first.y()[0][3], second.x()[0][0]);
        }
    }

    @Test
    void restoredStateRepeatsNextBatch() throws Exception {
        try (SequentialTextDataset dataset = dataset(8)) {
            dataset.nextBatch(2);
            long state = dataset.randomState();
            Batch expected = dataset.nextBatch(3);
            dataset.restoreRandomState(state);
            assertBatchEquals(expected, dataset.nextBatch(3));
        }
    }

    @Test
    void continuesAcrossBlocksAndWrapsAtEnd() throws Exception {
        try (SequentialTextDataset dataset = dataset(925)) {
            Batch first = dataset.nextBatch(1);
            dataset.nextBatch(1);
            Batch wrapped = dataset.nextBatch(1);
            assertEquals(first.x()[0][0], wrapped.x()[0][0]);
        }
    }

    @Test
    void rejectsInvalidArgumentsAndState() throws Exception {
        Path path = write(corpus());
        assertThrows(IllegalArgumentException.class,
                () -> new SequentialTextDataset(path, new ByteTokenizer(), 1, 1_024));
        try (SequentialTextDataset dataset = dataset(4)) {
            assertThrows(IllegalArgumentException.class, () -> dataset.nextBatch(0));
            assertThrows(IllegalArgumentException.class, () -> dataset.restoreRandomState(-1));
        }
    }

    private SequentialTextDataset dataset(int sequenceLength) throws Exception {
        return new SequentialTextDataset(write(corpus()), new ByteTokenizer(),
                sequenceLength, 1_024);
    }

    private static void assertBatchEquals(Batch expected, Batch actual) {
        for (int row = 0; row < expected.x().length; row++) {
            assertArrayEquals(expected.x()[row], actual.x()[row]);
            assertArrayEquals(expected.y()[row], actual.y()[row]);
        }
    }

    private Path write(String text) throws Exception {
        return Files.writeString(temporaryDirectory.resolve("corpus.txt"), text);
    }

    private static String corpus() {
        return "abcdefghijklmnopqrstuvwxyz0123456789\n".repeat(50);
    }
}
