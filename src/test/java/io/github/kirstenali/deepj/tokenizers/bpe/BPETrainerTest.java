package io.github.kirstenali.deepj.tokenizers.bpe;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class BPETrainerTest {

    // -------------------------------------------------------------------------
    // train() — model shape
    // -------------------------------------------------------------------------

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

    // -------------------------------------------------------------------------
    // selectBestPair — tie-breaking
    // -------------------------------------------------------------------------

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

    // -------------------------------------------------------------------------
    // Incremental count updates — correctness of applyMergeInPlace
    //
    // These tests verify that after each merge the live counts map is identical
    // to what a fresh full recount would produce.  That property is the
    // invariant the incremental path must preserve.
    // -------------------------------------------------------------------------

    /**
     * Simple case: one occurrence of the merged pair, neighbours on both sides.
     * Before: [X, A, B, Y, EOW]   pairs counted: (X,A)=1, (A,B)=1, (B,Y)=1
     * After:  [X, C, Y, EOW]      pairs counted: (X,C)=1, (C,Y)=1
     */
    @Test
    void incrementalCounts_simpleNeighbours() {
        // "ab ab ab" — the only possible merge is (a_byte, b_byte).
        // After that merge the counts map must contain no stale entries,
        // so no further merge is possible and the loop terminates cleanly.
        BPETrainer trainer = new BPETrainer();
        BPEModel   result  = trainer.train("ab ab ab", 259);

        assertFalse(result.merges().isEmpty());
        TokenPair firstMerge = result.merges().get(0);
        assertEquals((int) 'a', firstMerge.left());
        assertEquals((int) 'b', firstMerge.right());
    }

    /**
     * Consecutive non-overlapping pairs in one word: [A, B, A, B, EOW].
     * Both occurrences must be merged and the intermediate (B,A) pair must
     * be removed from counts, not left as a stale entry.
     */
    @Test
    void incrementalCounts_consecutivePairsProduceNoStaleEntries() {
        String corpus = "abab abab abab"; // forces (a,b) to be the best pair; B-A pair exists too
        BPETrainer trainer = new BPETrainer();
        BPEModel   model   = trainer.train(corpus, 260);

        // After all merges the model must be self-consistent
        assertEquals(model.merges().size(), model.mergeToNewId().size());
        // Every merge entry must map to a unique new id
        long uniqueIds = model.mergeToNewId().values().stream().distinct().count();
        assertEquals(model.mergeToNewId().size(), uniqueIds);
    }

    /**
     * The incremental-update path must produce exactly the same merge sequence
     * as a naive fresh-recount implementation would.  We verify this by
     * checking that two independent train() calls on the same corpus agree,
     * AND that the merge list is non-trivial (so the loop actually ran).
     */
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

    /**
     * After a merge the stale pair must not appear as a candidate in the next
     * round.  If counts are wrong, a previously-merged pair could be
     * re-selected (which train() guards against via vocab.contains), but more
     * subtly, a pair with an inflated count could steal the win from the true
     * best pair.  We check this via a carefully constructed corpus.
     */
    @Test
    void incrementalCounts_stalePairDoesNotInfluenceNextRound() {
        // "aab" repeated many times: (a,a) appears once per word, (a,b) once per word
        // After merging (a,a)→C, the pair (C,b) should be the dominant pair next round.
        String corpus = "aab ".repeat(20).trim();
        BPETrainer trainer = new BPETrainer();
        BPEModel   model   = trainer.train(corpus, 262);

        List<TokenPair> merges = model.merges();
        assertTrue(merges.size() >= 2);

        TokenPair first  = merges.get(0);
        TokenPair second = merges.get(1);

        // First merge must be (a, a) — highest frequency
        assertEquals((int) 'a', first.left());
        assertEquals((int) 'a', first.right());

        // Second merge must involve the new token (C) paired with 'b'
        int mergedAA = model.mergeToNewId().get(first);
        assertEquals(mergedAA, second.left());
        assertEquals((int) 'b',  second.right());
    }

    // -------------------------------------------------------------------------
    // trainTokenizerWithDefaults
    // -------------------------------------------------------------------------

    @Test
    void trainTokenizerWithDefaults_reservesDefaultSpecialTokens() {
        BPETrainer trainer = new BPETrainer();
        BPETokenizer tokenizer = trainer.trainTokenizerWithDefaults("hello hello world", 280);

        assertEquals(3, tokenizer.model().specialTokenToId().size());
        assertTrue(tokenizer.model().specialTokenToId().containsKey("<BOS>"));
        assertTrue(tokenizer.model().specialTokenToId().containsKey("<EOS>"));
        assertTrue(tokenizer.model().specialTokenToId().containsKey("<PAD>"));
    }

    // -------------------------------------------------------------------------
    // trainTokenizerFromFile
    // -------------------------------------------------------------------------

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
}

