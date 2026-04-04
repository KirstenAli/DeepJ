
package io.github.kirstenali.deepj.models.gpt;

import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class TextGeneratorTest {

    private Tokenizer tok;
    private GPTConfig cfg;
    private GPTModel model;

    @BeforeEach
    void setUp() {
        tok = new ByteTokenizer();
        cfg = new GPTConfig(
                ByteTokenizer.VOCAB_SIZE,
                16,   // maxSeqLen
                32,   // dModel
                4,    // nHeads
                2,    // nLayers
                64    // dFF
        );
        model = new GPTModel(cfg, 1L);
    }

    // ── basic smoke test ───────────────────────────────────────────

    @Test
    void generate_runsAndReturnsNonEmptyString() {
        String out = TextGenerator.generate(model, tok, cfg, "hi", 8, 1.0, 0, 2L);

        assertNotNull(out);
        assertTrue(out.length() >= 2, "output must include at least the prompt");
    }

    // ── determinism ────────────────────────────────────────────────

    @Test
    void sameSeedProducesSameOutput() {
        String a = TextGenerator.generate(model, tok, cfg, "hello", 10, 0.8, 5, 42L);
        String b = TextGenerator.generate(model, tok, cfg, "hello", 10, 0.8, 5, 42L);

        assertEquals(a, b, "identical seeds must produce identical output");
    }

    @Test
    void differentSeedsProduceDifferentOutput() {
        String a = TextGenerator.generate(model, tok, cfg, "hello", 20, 1.0, 0, 1L);
        String b = TextGenerator.generate(model, tok, cfg, "hello", 20, 1.0, 0, 999L);

        // Not guaranteed but overwhelmingly likely with 20 tokens
        assertNotEquals(a, b, "different seeds should usually diverge");
    }

    // ── prompt preservation ────────────────────────────────────────

    @Test
    void outputStartsWithPrompt() {
        String prompt = "abc";
        String out = TextGenerator.generate(model, tok, cfg, prompt, 5, 1.0, 0, 7L);

        assertTrue(out.startsWith(prompt), "output must begin with the prompt");
    }

    // ── maxNewTokens edge cases ────────────────────────────────────

    @Test
    void zeroNewTokensReturnsPromptOnly() {
        String prompt = "test";
        String out = TextGenerator.generate(model, tok, cfg, prompt, 0, 1.0, 0, 1L);

        assertEquals(prompt, out, "zero new tokens should return the prompt unchanged");
    }

    @Test
    void outputGrowsWithMoreTokens() {
        String short_ = TextGenerator.generate(model, tok, cfg, "x", 2, 1.0, 0, 5L);
        String long_  = TextGenerator.generate(model, tok, cfg, "x", 20, 1.0, 0, 5L);

        assertTrue(long_.length() > short_.length(), "more tokens should produce longer output");
    }

    // ── topK behaviour ─────────────────────────────────────────────

    @Test
    void topKOneIsGreedy() {
        // topK=1 always picks the highest-probability token → deterministic regardless of seed
        String a = TextGenerator.generate(model, tok, cfg, "hi", 10, 1.0, 1, 1L);
        String b = TextGenerator.generate(model, tok, cfg, "hi", 10, 1.0, 1, 999L);

        assertEquals(a, b, "topK=1 should be greedy and seed-independent");
    }

    @Test
    void topKZeroUsesFullVocab() {
        // Should not throw — topK=0 means "use all logits"
        String out = TextGenerator.generate(model, tok, cfg, "hi", 5, 1.0, 0, 1L);
        assertNotNull(out);
    }


    // ── validation ─────────────────────────────────────────────────

    @Test
    void negativeMaxNewTokensThrows() {
        assertThrows(IllegalArgumentException.class,
                () -> TextGenerator.generate(model, tok, cfg, "x", -1, 1.0, 0, 1L));
    }

    @Test
    void zeroTemperatureThrows() {
        assertThrows(IllegalArgumentException.class,
                () -> TextGenerator.generate(model, tok, cfg, "x", 5, 0.0, 0, 1L));
    }

    @Test
    void negativeTemperatureThrows() {
        assertThrows(IllegalArgumentException.class,
                () -> TextGenerator.generate(model, tok, cfg, "x", 5, -0.5, 0, 1L));
    }

    @Test
    void negativeTopKThrows() {
        assertThrows(IllegalArgumentException.class,
                () -> TextGenerator.generate(model, tok, cfg, "x", 5, 1.0, -1, 1L));
    }
}
