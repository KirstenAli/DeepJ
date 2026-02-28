package org.deepj.ann.tokenizer;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class ByteTokenizerTest {

    @Test
    public void roundTripEncodeDecode() {
        ByteTokenizer tok = new ByteTokenizer();
        String text = "Hello, world! ✅ — café";
        int[] ids = tok.encode(text);
        String decoded = tok.decode(ids);
        assertEquals(text, decoded);
    }
}
