package io.github.kirstenali.deepj.data;

import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class IndexedResponseTextDatasetTest {

    @TempDir
    Path temporaryDirectory;

    @Test
    void indexesAndMasksResponseRecords() throws Exception {
        try (var dataset = dataset(records("Hello", "Hi") + records("Question", "Answer"))) {
            Batch batch = dataset.nextBatch(1);

            assertEquals(2, dataset.recordCount());
            assertFalse(batch.mask(0)[0]);
            assertTrue(batch.mask(0)[batch.mask(0).length - 1]);
        }
    }

    @Test
    void restoresSamplingState() throws Exception {
        try (var dataset = dataset(records("One", "First") + records("Two", "Second"))) {
            long state = dataset.randomState();
            Batch expected = dataset.nextBatch(1);
            dataset.restoreRandomState(state);
            Batch actual = dataset.nextBatch(1);

            assertArrayEquals(expected.x()[0], actual.x()[0]);
            assertArrayEquals(expected.y()[0], actual.y()[0]);
            assertArrayEquals(expected.mask(0), actual.mask(0));
        }
    }

    private IndexedResponseTextDataset dataset(String text) throws Exception {
        Path path = Files.writeString(temporaryDirectory.resolve("records.txt"), text);
        var source = new ResponseOnlyTextDataset.Source(path, 1);
        return new IndexedResponseTextDataset(List.of(source), new ByteTokenizer(), 256, 7L);
    }

    private static String records(String instruction, String response) {
        return "Instruction:\n" + instruction + "\nResponse:\n" + response
                + "\n<|endoftext|>\n\n";
    }
}
