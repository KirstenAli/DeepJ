package io.github.kirstenali.deepj.tokenizers.bpe;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
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
    void train_skipsDuplicateMergeCandidatesAndContinues() {
        BPETrainer trainer = new BPETrainer();
        BPEModel model = trainer.train("aaaaaa", 260);

        assertTrue(model.vocabSize() > 257);
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
    void incrementalCounts_simpleNeighbours() {

        BPETrainer trainer = new BPETrainer();
        BPEModel   result  = trainer.train("ab ab ab", 259);

        assertFalse(result.merges().isEmpty());
        TokenPair firstMerge = result.merges().get(0);
        assertEquals((int) 'a', firstMerge.left());
        assertEquals((int) 'b', firstMerge.right());
    }

    @Test
    void incrementalCounts_consecutivePairsProduceNoStaleEntries() {
        String corpus = "abab abab abab";
        BPETrainer trainer = new BPETrainer();
        BPEModel   model   = trainer.train(corpus, 260);

        assertEquals(model.merges().size(), model.mergeToNewId().size());

        long uniqueIds = model.mergeToNewId().values().stream().distinct().count();
        assertEquals(model.mergeToNewId().size(), uniqueIds);
    }

    @Test
    void incrementalCounts_mergeSequenceMatchesFreshRecount() {
        String corpus = "the cat sat on the mat the cat sat";
        BPETrainer trainer = new BPETrainer();

        BPEModel a = trainer.train(corpus, 290);
        BPEModel b = trainer.train(corpus, 290);

        assertEquals(a.merges(),      b.merges());
        assertEquals(a.mergeToNewId(), b.mergeToNewId());
        assertTrue(a.merges().size() >= 2, "expected multiple merge rounds on this corpus");
    }

    @Test
    void incrementalCounts_stalePairDoesNotInfluenceNextRound() {

        String corpus = "aab ".repeat(20).trim();
        BPETrainer trainer = new BPETrainer();
        BPEModel   model   = trainer.train(corpus, 262);

        List<TokenPair> merges = model.merges();
        assertTrue(merges.size() >= 2);

        TokenPair first  = merges.get(0);
        TokenPair second = merges.get(1);

        assertEquals((int) 'a', first.left());
        assertEquals((int) 'a', first.right());

        int mergedAA = model.mergeToNewId().get(first);
        assertEquals(mergedAA, second.left());
        assertEquals((int) 'b',  second.right());
    }

    @Test
    void incrementalCounts_keepsDecreasedPairsEligible() {
        BPEModel model = new BPETrainer().train("aaaababab", 259);

        assertEquals(List.of(
                new TokenPair('a', 'a'),
                new TokenPair('a', 'b')
        ), model.merges());
    }

    @Test
    void trainTokenizerWithDefaults_reservesDefaultSpecialTokens() {
        BPETrainer trainer = new BPETrainer();
        BPETokenizer tokenizer = trainer.trainTokenizerWithDefaults("hello hello world", 280);

        assertEquals(3, tokenizer.model().specialTokenToId().size());
        assertTrue(tokenizer.model().specialTokenToId().containsKey("<BOS>"));
        assertTrue(tokenizer.model().specialTokenToId().containsKey("<EOS>"));
        assertTrue(tokenizer.model().specialTokenToId().containsKey("<PAD>"));
    }

    @Test
    void trainTokenizerFromFile_supportsBothOverloads() throws IOException {
        Path temp = Files.createTempFile("deepj-bpe-train", ".txt");
        try {
            Files.writeString(temp, "hello hello world");
            BPETrainer trainer = new BPETrainer();

            BPETokenizer plain       = trainer.trainTokenizerFromFile(temp, 270);
            BPETokenizer withSpecials = trainer.trainTokenizerFromFile(temp, 280, List.of("<BOS>", "<EOS>", "<PAD>"));

            assertNotNull(plain);
            assertNotNull(withSpecials);
            assertTrue(withSpecials.model().specialTokenToId().containsKey("<BOS>"));
        } finally {
            Files.deleteIfExists(temp);
        }
    }

    @Test
    void trainFromFile_supportsBoundedSamples() throws IOException {
        Path temp = Files.createTempFile("deepj-bpe-sample", ".txt");
        Files.writeString(temp, "aaaaa zzzzz");
        BPETrainer trainer = new BPETrainer();
        assertNotNull(trainer.trainFromFile(temp, 260, List.of(), 5));
        assertThrows(IllegalArgumentException.class,
                () -> trainer.trainFromFile(temp, 260, List.of(), 0));
    }
}
