package io.github.kirstenali.deepj.tokenizers.bpe;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class BPETokenizerTest {

    @Test
    void encodeDecode_roundTripsAscii() {
        String trainingText = "low lower lowest low lower lowest";
        String input = "low lowest";

        BPETrainer trainer = new BPETrainer();
        BPETokenizer tokenizer = trainer.trainTokenizer(trainingText, 280);

        int[] ids = tokenizer.encode(input);
        String decoded = tokenizer.decode(ids);

        assertEquals(input, decoded);
    }

    @Test
    void encodeDecode_roundTripsWhitespaceExactly() {
        String trainingText = "hello   world\nhello   world\n";
        String input = "hello   world\n";

        BPETrainer trainer = new BPETrainer();
        BPETokenizer tokenizer = trainer.trainTokenizer(trainingText, 280);

        int[] ids = tokenizer.encode(input);
        String decoded = tokenizer.decode(ids);

        assertEquals(input, decoded);
    }

    @Test
    void encodeDecode_roundTripsUnicode() {
        String trainingText = "héllo héllo café café 🚀🚀";
        String input = "héllo café 🚀";

        BPETrainer trainer = new BPETrainer();
        BPETokenizer tokenizer = trainer.trainTokenizer(trainingText, 290);

        int[] ids = tokenizer.encode(input);
        String decoded = tokenizer.decode(ids);

        assertEquals(input, decoded);
    }

    @Test
    void encodingAfterTrainingUsuallyUsesFewerTokensForFrequentPatterns() {
        String trainingText = "banana banana banana banana banana";
        String input = "banana";

        BPETrainer trainer = new BPETrainer();
        BPETokenizer tokenizer = trainer.trainTokenizer(trainingText, 270);

        int[] ids = tokenizer.encode(input);

        assertTrue(ids.length < input.getBytes().length,
                "Expected trained tokenizer to compress frequent pattern into fewer tokens");
    }

    @Test
    void decode_rejectsInvalidTokenIds() {
        BPETrainer trainer = new BPETrainer();
        BPETokenizer tokenizer = trainer.trainTokenizer("hello hello hello", 270);

        assertThrows(IllegalArgumentException.class, () -> tokenizer.decode(new int[]{999999}));
    }

    @Test
    void encode_emptyStringProducesNoTokens() {
        BPETrainer trainer = new BPETrainer();
        BPETokenizer tokenizer = trainer.trainTokenizer("hello world", 270);

        int[] ids = tokenizer.encode("");

        assertArrayEquals(new int[0], ids);
    }
}