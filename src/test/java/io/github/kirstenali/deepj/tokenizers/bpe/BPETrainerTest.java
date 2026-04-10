package io.github.kirstenali.deepj.tokenizers.bpe;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class BPETrainerTest {

    @Test
    void train_createsModel() {
        BPETrainer trainer = new BPETrainer();
        BPEModel model = trainer.train("banana banana banana", 270);

        assertNotNull(model);
        assertTrue(model.vocabSize() > 257);
        assertEquals(model.merges().size(), model.mergeToNewId().size());
    }

    @Test
    void train_respectsTargetUpperBound() {
        BPETrainer trainer = new BPETrainer();
        BPEModel model = trainer.train("banana banana banana", 265);

        assertTrue(model.vocabSize() <= 265);
    }

    @Test
    void train_rejectsTooSmallTargetVocab() {
        BPETrainer trainer = new BPETrainer();

        IllegalArgumentException ex = assertThrows(
                IllegalArgumentException.class,
                () -> trainer.train("hello", 257)
        );

        assertTrue(ex.getMessage().contains("targetVocabSize"));
    }

    @Test
    void train_isDeterministic() {
        String text = "banana bandana banana bandana";
        BPETrainer trainer = new BPETrainer();

        BPEModel m1 = trainer.train(text, 270);
        BPEModel m2 = trainer.train(text, 270);

        assertEquals(m1.merges(), m2.merges());
        assertEquals(m1.mergeToNewId(), m2.mergeToNewId());
        assertEquals(m1.vocabSize(), m2.vocabSize());
    }

    @Test
    void selectBestPair_breaksTiesBySmallestPair() {
        Map<TokenPair, Integer> counts = Map.of(
                new TokenPair(1, 2), 3,
                new TokenPair(2, 3), 5,
                new TokenPair(3, 4), 5
        );

        TokenPair best = BPETrainer.selectBestPair(counts);

        assertEquals(new TokenPair(2, 3), best);
    }

    @Test
    void train_skipsDuplicateMergeCandidatesAndContinues() {
        BPETrainer trainer = new BPETrainer();

        BPEModel model = trainer.train("aaaaaa", 260);

        assertTrue(model.vocabSize() > 257);
    }

    @Test
    void trainProductionTokenizer_reservesDefaultSpecialTokens() {
        BPETrainer trainer = new BPETrainer();
        BPETokenizer tokenizer = trainer.trainProductionTokenizer("hello hello world", 280);

        assertEquals(3, tokenizer.model().specialTokenToId().size());
        assertTrue(tokenizer.model().specialTokenToId().containsKey("<BOS>"));
        assertTrue(tokenizer.model().specialTokenToId().containsKey("<EOS>"));
        assertTrue(tokenizer.model().specialTokenToId().containsKey("<PAD>"));
    }
}