package io.github.kirstenali.deepj.examples;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PrepareKnowledgeCorpusTest {

    private static final String HEADER = "Below is an instruction that describes a task. "
            + "Write a response that appropriately completes the request.";

    @TempDir
    Path tempDir;

    @Test
    void preparesSeparateTrainingAndValidationData() throws Exception {
        Path alpaca = Files.writeString(tempDir.resolve("alpaca.txt"), alpacaRecords());
        Path stories = Files.writeString(tempDir.resolve("stories.txt"), storyRecords());
        Path output = tempDir.resolve("output");

        var result = PrepareKnowledgeCorpus.prepare(alpaca, stories, output, 1, 1, 2);
        String training = Files.readString(output.resolve("knowledge-train.txt"));
        String validation = Files.readString(output.resolve("knowledge-valid.txt"));

        assertSplitCounts(result);
        assertCorpusContents(training, validation);
    }

    private static void assertSplitCounts(PrepareKnowledgeCorpus.PreparationResult result) {
        assertEquals(20, result.alpacaSplit().training());
        assertEquals(2, result.alpacaSplit().validation());
        assertEquals(68, result.factSplit().training());
        assertEquals(4, result.factSplit().validation());
    }

    private static void assertCorpusContents(String training, String validation) {
        assertTrue(training.contains("What is 1 plus 1?"));
        assertEquals(2, training.lines().filter("What is 1 plus 1?"::equals).count());
        assertTrue(training.contains("Story one."));
        assertTrue(validation.contains("Question 0"));
        assertTrue(validation.contains("Question 20"));
        assertTrue(validation.contains("What is 0 plus 0?"));
    }

    private static String alpacaRecords() {
        StringBuilder text = new StringBuilder();
        for (int index = 0; index < 22; index++) {
            text.append(HEADER).append("\n\nQuestion ").append(index)
                    .append("\n\nAnswer ").append(index).append("\n\n");
        }
        return text.toString();
    }

    private static String storyRecords() {
        return "Story one.\n<|endoftext|>\nStory two.\n<|endoftext|>\n";
    }
}
