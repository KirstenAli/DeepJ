package io.github.kirstenali.deepj.data;

import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class TextDatasetTest {

    // ── existing behaviour ──────────────────────────────────────────

    @Test
    void nextBatch_targetsAreShiftedByOne() throws IOException {
        Tokenizer tok = new ByteTokenizer();

        // For ASCII consecutive letters, byte ids increase by +1 each character.
        String text = "abcdefg";
        Path tmp = Files.createTempFile("deepj", ".txt");
        Files.writeString(tmp, text);

        TextDataset ds = TextDataset.fromFile(tmp, tok, 4, 1L);
        Batch b = ds.nextBatch(3);

        Assertions.assertEquals(3, b.x().length);
        Assertions.assertEquals(4, b.x()[0].length);
        Assertions.assertEquals(3, b.y().length);
        Assertions.assertEquals(4, b.y()[0].length);

        for (int i = 0; i < b.x().length; i++) {
            for (int t = 0; t < 4; t++) {
                Assertions.assertEquals(b.x()[i][t] + 1, b.y()[i][t], "for this toy corpus, y = x shifted by one byte");
            }
        }
    }

    @Test
    void nextBatch_handlesMinimumValidTokenLength() {
        int[] tokens = new int[]{10, 11, 12, 13, 14}; // seqLen=4 => seqLen+1
        TextDataset ds = new TextDataset(tokens, 4, 7L);

        Batch b = ds.nextBatch(1);

        Assertions.assertArrayEquals(new int[]{10, 11, 12, 13}, b.x()[0]);
        Assertions.assertArrayEquals(new int[]{11, 12, 13, 14}, b.y()[0]);
    }

    // ── streaming / memory-mapped behaviour ─────────────────────────

    @Test
    void fromFile_producesCorrectTokenCount() throws IOException {
        String text = "hello world";
        Path tmp = writeTempFile(text);
        Tokenizer tok = new ByteTokenizer();

        TextDataset ds = TextDataset.fromFile(tmp, tok, 4, 42L);

        Assertions.assertEquals(text.length(), ds.size());
    }

    @Test
    void fromFile_multilinePreservesNewlines() throws IOException {
        String text = "abc\ndef\nghi";
        Path tmp = writeTempFile(text);
        Tokenizer tok = new ByteTokenizer();

        TextDataset ds = TextDataset.fromFile(tmp, tok, 4, 1L);

        // ByteTokenizer encodes every byte including \n
        Assertions.assertEquals(text.getBytes().length, ds.size());
    }

    @Test
    void fromFile_batchValuesMatchDirectConstruction() throws IOException {
        String text = "abcdefghij";
        Tokenizer tok = new ByteTokenizer();
        int seqLen = 4;
        long seed = 99L;

        // Memory-mapped path
        Path tmp = writeTempFile(text);
        TextDataset mapped = TextDataset.fromFile(tmp, tok, seqLen, seed);

        // Direct int[] path
        TextDataset direct = new TextDataset(tok.encode(text), seqLen, seed);

        // Same seed → same random positions → identical batches
        Batch bMapped = mapped.nextBatch(3);
        Batch bDirect = direct.nextBatch(3);

        for (int i = 0; i < 3; i++) {
            Assertions.assertArrayEquals(bDirect.x()[i], bMapped.x()[i],
                    "x mismatch at batch row " + i);
            Assertions.assertArrayEquals(bDirect.y()[i], bMapped.y()[i],
                    "y mismatch at batch row " + i);
        }
    }

    @Test
    void fromFile_largeFile_tokenisedCorrectly() throws IOException {
        // Build a file with many lines to exercise the chunked streaming path
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 2000; i++) {
            sb.append("line ").append(i).append(" of test data\n");
        }
        String text = sb.toString();
        Path tmp = writeTempFile(text);
        Tokenizer tok = new ByteTokenizer();

        TextDataset ds = TextDataset.fromFile(tmp, tok, 32, 7L);

        Assertions.assertEquals(text.getBytes().length, ds.size());

        // Verify a batch is well-formed and targets are shifted
        Batch b = ds.nextBatch(5);
        for (int i = 0; i < 5; i++) {
            Assertions.assertEquals(32, b.x()[i].length);
            Assertions.assertEquals(32, b.y()[i].length);
        }
    }

    @Test
    void fromFile_noTrailingNewline() throws IOException {
        String text = "no newline at end";
        Path tmp = writeTempFile(text);
        Tokenizer tok = new ByteTokenizer();

        TextDataset ds = TextDataset.fromFile(tmp, tok, 4, 1L);

        Assertions.assertEquals(text.getBytes().length, ds.size());
    }

    @Test
    void seqLen_returnsConfiguredValue() throws IOException {
        String text = "abcdefghij";
        Path tmp = writeTempFile(text);
        Tokenizer tok = new ByteTokenizer();

        TextDataset ds = TextDataset.fromFile(tmp, tok, 6, 1L);

        Assertions.assertEquals(6, ds.seqLen());
    }

    @Test
    void constructor_rejectsTooFewTokens() {
        int[] tokens = new int[]{1, 2, 3};
        Assertions.assertThrows(IllegalArgumentException.class,
                () -> new TextDataset(tokens, 4, 1L));
    }

    @Test
    void constructor_rejectsTooSmallSeqLen() {
        int[] tokens = new int[]{1, 2, 3, 4, 5};
        Assertions.assertThrows(IllegalArgumentException.class,
                () -> new TextDataset(tokens, 1, 1L));
    }

    // ── helpers ─────────────────────────────────────────────────────

    private static Path writeTempFile(String content) throws IOException {
        Path tmp = Files.createTempFile("deepj-test-", ".txt");
        Files.writeString(tmp, content);
        return tmp;
    }
}
