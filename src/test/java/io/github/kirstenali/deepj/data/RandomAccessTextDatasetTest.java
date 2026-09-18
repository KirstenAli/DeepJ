package io.github.kirstenali.deepj.data;

import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETokenizer;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETrainer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RandomAccessTextDatasetTest {

    @TempDir
    Path temporaryDirectory;

    @Test
    void batchesHaveShiftedTargets() throws Exception {
        Path corpus = write("abcdefghijklmnopqrstuvwxyz\n".repeat(500));
        try (RandomAccessTextDataset dataset =
                     new RandomAccessTextDataset(corpus, new ByteTokenizer(), 32, 7L)) {
            Batch batch = dataset.nextBatch(3);
            for (int row = 0; row < 3; row++) {
                for (int token = 0; token < 31; token++) {
                    assertEquals(batch.x()[row][token + 1], batch.y()[row][token]);
                }
            }
        }
    }

    @Test
    void completeFilePreservesSpecialToken() throws Exception {
        String text = "one story\n<|endoftext|>\ntwo stories\n";
        List<String> specials = List.of("<|endoftext|>");
        BPETokenizer tokenizer = new BPETrainer().trainTokenizer(text.repeat(10), 270, specials);
        int[] expected = tokenizer.encode(text);
        int batchSize = (expected.length - 1) / 2;
        try (RandomAccessTextDataset dataset =
                     new RandomAccessTextDataset(write(text), tokenizer, 2, 1L)) {
            Batch batch = dataset.nextBatch(batchSize);
            assertTrue(contains(batch, tokenizer.model().specialTokenToId().get("<|endoftext|>")));
        }
    }

    @Test
    void rejectsEmptyFilesAndInvalidArguments() throws Exception {
        Path empty = write("");
        assertThrows(IllegalArgumentException.class,
                () -> new RandomAccessTextDataset(empty, new ByteTokenizer(), 8, 1L));
        Path content = write("some content");
        try (RandomAccessTextDataset dataset =
                     new RandomAccessTextDataset(content, new ByteTokenizer(), 2, 1L)) {
            assertThrows(IllegalArgumentException.class, () -> dataset.nextBatch(0));
        }
    }

    private Path write(String text) throws Exception {
        Path path = temporaryDirectory.resolve("corpus-" + Math.abs(text.hashCode()) + ".txt");
        return Files.writeString(path, text);
    }

    private static boolean contains(Batch batch, int tokenId) {
        for (int[] row : batch.x()) {
            for (int token : row) if (token == tokenId) return true;
        }
        for (int token : batch.y()[batch.y().length - 1]) if (token == tokenId) return true;
        return false;
    }
}
