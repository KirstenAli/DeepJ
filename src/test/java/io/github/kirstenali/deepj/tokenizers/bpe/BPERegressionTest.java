package io.github.kirstenali.deepj.tokenizers.bpe;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class BPERegressionTest {

    @Test
    void byteZeroStillExistsInVocabulary() {
        BPETrainer trainer = new BPETrainer();
        BPEModel model = trainer.train("abc abc abc", 270);

        Integer zeroId = model.bytesToId().get(BPEBytes.key(new byte[]{0}));
        assertNotNull(zeroId);
        assertEquals(0, zeroId);
    }

    @Test
    void decodingDoesNotInsertNullCharacters() {
        String text = "banana banana banana banana";
        BPETrainer trainer = new BPETrainer();
        BPETokenizer tokenizer = trainer.trainTokenizer(text, 280);

        String decoded = tokenizer.decode(tokenizer.encode("banana"));
        assertEquals("banana", decoded);
        assertFalse(decoded.contains("\u0000"));
    }
}