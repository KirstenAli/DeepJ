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

class ResponseOnlyTextDatasetTest {

    @TempDir
    Path temporaryDirectory;

    @Test
    void masksPromptAndLearnsResponse() throws Exception {
        ResponseOnlyTextDataset dataset = dataset(records("Question one", "Answer one"));
        var example = dataset.examples().get(0);
        boolean[] mask = dataset.nextBatch(1).mask(0);
        assertEquals(example.tokens().length - 1, mask.length);
        assertMaskBoundary(mask, example.responseStart() - 1);
    }

    @Test
    void preservesPromptAndResponseText() throws Exception {
        ResponseOnlyTextDataset dataset = dataset(records("Question one", "Answer one"));
        var example = dataset.examples().get(0);
        assertEquals("Instruction:\nQuestion one\nResponse:\n", example.prompt());
        assertEquals("Answer one", example.response());
    }

    @Test
    void restoredStateRepeatsBatch() throws Exception {
        ResponseOnlyTextDataset dataset = dataset(records("One", "First") + records("Two", "Second"));
        dataset.nextBatch(1);
        long state = dataset.randomState();
        Batch expected = dataset.nextBatch(2);
        dataset.restoreRandomState(state);
        assertBatchEquals(expected, dataset.nextBatch(2));
    }

    private ResponseOnlyTextDataset dataset(String text) throws Exception {
        Path path = Files.writeString(temporaryDirectory.resolve("records.txt"), text);
        var source = new ResponseOnlyTextDataset.Source(path, 1);
        return new ResponseOnlyTextDataset(List.of(source), new ByteTokenizer(), 256, 7L);
    }

    private static String records(String instruction, String response) {
        return "Instruction:\n" + instruction + "\nResponse:\n" + response
                + "\n<|endoftext|>\n\n";
    }

    private static void assertMaskBoundary(boolean[] mask, int boundary) {
        for (int index = 0; index < boundary; index++) assertFalse(mask[index]);
        for (int index = boundary; index < mask.length; index++) assertTrue(mask[index]);
    }

    private static void assertBatchEquals(Batch expected, Batch actual) {
        for (int row = 0; row < expected.x().length; row++) {
            assertArrayEquals(expected.x()[row], actual.x()[row]);
            assertArrayEquals(expected.y()[row], actual.y()[row]);
            assertArrayEquals(expected.lossMask()[row], actual.lossMask()[row]);
        }
    }
}
