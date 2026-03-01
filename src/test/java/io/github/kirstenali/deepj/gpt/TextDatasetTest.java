package io.github.kirstenali.deepj.gpt;

import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class TextDatasetTest {

    @Test
    public void batchHasCorrectShapesAndShift() {
        Tokenizer tok = new ByteTokenizer();
        int[] tokens = tok.encode("abcdefg");
        TextDataset ds = new TextDataset(tokens, 4, 123);

        Batch b = ds.nextBatch(2);

        assertEquals(2, b.x().length);
        assertEquals(2, b.y().length);
        assertEquals(4, b.x()[0].length);
        assertEquals(4, b.y()[0].length);

        for (int i = 0; i < 2; i++) {
            // y is x shifted by 1 in source stream, so within each sample y[t] should be the next token after x[t]
            // Not strictly equal to x[t+1] because sampling is contiguous; within a chunk it is.
            for (int t = 0; t < 3; t++) {
                assertEquals(b.x()[i][t+1], b.y()[i][t]);
            }
        }
    }
}
