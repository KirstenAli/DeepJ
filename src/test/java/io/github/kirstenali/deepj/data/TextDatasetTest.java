package io.github.kirstenali.deepj.data;

import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

public class TextDatasetTest {

    @Test
    void nextBatch_targetsAreShiftedByOne() throws IOException {
        Tokenizer tok = new ByteTokenizer();

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
    void nextBatch_handlesMinimumValidTokenLength() throws IOException {
        int[] tokens = new int[]{10, 11, 12, 13, 14};
        TextDataset ds = fromTokens(tokens, 4, 7L);

        Batch b = ds.nextBatch(1);

        Assertions.assertArrayEquals(new int[]{10, 11, 12, 13}, b.x()[0]);
        Assertions.assertArrayEquals(new int[]{11, 12, 13, 14}, b.y()[0]);
    }

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

        Assertions.assertEquals(text.getBytes().length, ds.size());
    }

    @Test
    void fromFile_batchValuesMatchDirectConstruction() throws IOException {
        String text = "abcdefghij";
        Tokenizer tok = new ByteTokenizer();
        int seqLen = 4;
        long seed = 99L;

        Path tmp = writeTempFile(text);
        TextDataset mapped = TextDataset.fromFile(tmp, tok, seqLen, seed);

        TextDataset direct = fromTokens(tok.encode(text), seqLen, seed);

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

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 2000; i++) {
            sb.append("line ").append(i).append(" of test data\n");
        }
        String text = sb.toString();
        Path tmp = writeTempFile(text);
        Tokenizer tok = new ByteTokenizer();

        TextDataset ds = TextDataset.fromFile(tmp, tok, 32, 7L);

        Assertions.assertEquals(text.getBytes().length, ds.size());

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
                () -> fromTokens(tokens, 4, 1L));
    }

    @Test
    void constructor_rejectsTooSmallSeqLen() {
        int[] tokens = new int[]{1, 2, 3, 4, 5};
        Assertions.assertThrows(IllegalArgumentException.class,
                () -> fromTokens(tokens, 1, 1L));
    }

    @Test
    void fromBinaryFile_rejectsPartialToken() throws IOException {
        Path tmp = Files.createTempFile("deepj-invalid-tokens-", ".bin");
        Files.write(tmp, new byte[Integer.BYTES + 1]);

        Assertions.assertThrows(IOException.class,
                () -> TextDataset.fromBinaryFile(tmp, 2, 1L));
    }

    @Test
    void fromBinaryFile_rejectsInvalidChunkSize() throws IOException {
        int[] tokens = new int[]{1, 2, 3, 4, 5};
        Path tmp = writeTokenFile(tokens);

        Assertions.assertThrows(IllegalArgumentException.class,
                () -> TextDataset.fromBinaryFile(tmp, 2, 1L, 3L));
    }

    @Test
    void nextBatch_rejectsEmptyBatch() throws IOException {
        TextDataset ds = fromTokens(new int[]{1, 2, 3, 4, 5}, 2, 1L);

        Assertions.assertThrows(IllegalArgumentException.class, () -> ds.nextBatch(0));
    }

    @Test
    void fromFile_multiSegment_readsAcrossChunkBoundary() throws IOException {

        String text = "abcdefghijkl";
        Path tmp = writeTempFile(text);
        Tokenizer tok = new ByteTokenizer();
        int seqLen = 4;
        long seed = 42L;

        TextDataset single = TextDataset.fromFile(tmp, tok, seqLen, seed);

        TextDataset multi  = TextDataset.fromFile(tmp, tok, seqLen, seed, 20L);

        Assertions.assertEquals(single.size(), multi.size());

        Batch bs = single.nextBatch(4);
        Batch bm = multi.nextBatch(4);
        for (int i = 0; i < 4; i++) {
            Assertions.assertArrayEquals(bs.x()[i], bm.x()[i], "x mismatch at row " + i);
            Assertions.assertArrayEquals(bs.y()[i], bm.y()[i], "y mismatch at row " + i);
        }
    }

    private static TextDataset fromTokens(int[] tokens, int seqLen, long seed) throws IOException {
        return TextDataset.fromBinaryFile(writeTokenFile(tokens), seqLen, seed);
    }

    private static Path writeTokenFile(int[] tokens) throws IOException {
        Path tmp = Files.createTempFile("deepj-test-tokens-", ".bin");
        ByteBuffer buf = ByteBuffer.allocate(tokens.length * Integer.BYTES);
        for (int t : tokens) {
            buf.putInt(t);
        }
        buf.flip();
        try (FileChannel ch = FileChannel.open(tmp, StandardOpenOption.WRITE)) {
            ch.write(buf);
        }
        return tmp;
    }

    private static Path writeTempFile(String content) throws IOException {
        Path tmp = Files.createTempFile("deepj-test-", ".txt");
        Files.writeString(tmp, content);
        return tmp;
    }
}
