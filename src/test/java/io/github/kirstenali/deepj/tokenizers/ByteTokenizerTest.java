
package io.github.kirstenali.deepj.tokenizers;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class ByteTokenizerTest {

    @Test
    void roundTrip_utf8() {
        Tokenizer tok = new ByteTokenizer();
        String text = "Hello, 世界 🌍";
        int[] ids = tok.encode(text);
        String decoded = tok.decode(ids);
        Assertions.assertEquals(text, decoded);
    }

    @Test
    void decode_rejectsOutOfRangeIds() {
        Tokenizer tok = new ByteTokenizer();
        Assertions.assertThrows(IllegalArgumentException.class, () -> tok.decode(new int[]{256}));
        Assertions.assertThrows(IllegalArgumentException.class, () -> tok.decode(new int[]{-1}));
    }

    @Test
    void vocabSize_is256() {
        Assertions.assertEquals(256, ByteTokenizer.VOCAB_SIZE);
    }
}
