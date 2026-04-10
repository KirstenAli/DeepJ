
package io.github.kirstenali.deepj.data;

import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class TextDatasetTest {

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
}
