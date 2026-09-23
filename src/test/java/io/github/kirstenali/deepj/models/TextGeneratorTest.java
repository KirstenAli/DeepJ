package io.github.kirstenali.deepj.models;

import io.github.kirstenali.deepj.models.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.models.origin.DeepJOrigin;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

public class TextGeneratorTest {

    private Tokenizer tok;
    private DeepJOriginConfig cfg;
    private DeepJOrigin model;

    @BeforeEach
    void setUp() {
        tok = new ByteTokenizer();
        cfg = new DeepJOriginConfig(
                ByteTokenizer.VOCAB_SIZE,
                16,
                32,
                4,
                2,
                64
        );
        model = new DeepJOrigin(cfg, 1L);
    }

    @Test
    void generate_runsAndReturnsNonEmptyString() {
        String out = TextGenerator.generate(model, tok, cfg, "hi", 8, 1.0f, 0, 2L);

        assertNotNull(out);
        assertTrue(out.length() >= 2, "output must include at least the prompt");
    }

    @Test
    void sameSeedProducesSameOutput() {
        String a = TextGenerator.generate(model, tok, cfg, "hello", 10, 0.8f, 5, 42L);
        String b = TextGenerator.generate(model, tok, cfg, "hello", 10, 0.8f, 5, 42L);

        assertEquals(a, b, "identical seeds must produce identical output");
    }

    @Test
    void differentSeedsProduceDifferentOutput() {
        String a = TextGenerator.generate(model, tok, cfg, "hello", 20, 1.0f, 0, 1L);
        String b = TextGenerator.generate(model, tok, cfg, "hello", 20, 1.0f, 0, 999L);

        assertNotEquals(a, b, "different seeds should usually diverge");
    }

    @Test
    void outputStartsWithPrompt() {
        String prompt = "abc";
        String out = TextGenerator.generate(model, tok, cfg, prompt, 5, 1.0f, 0, 7L);

        assertTrue(out.startsWith(prompt), "output must begin with the prompt");
    }

    @Test
    void zeroNewTokensReturnsPromptOnly() {
        String prompt = "test";
        String out = TextGenerator.generate(model, tok, cfg, prompt, 0, 1.0f, 0, 1L);

        assertEquals(prompt, out, "zero new tokens should return the prompt unchanged");
    }

    @Test
    void generationStopsBeforeEndOfSequenceToken() {
        AtomicInteger calls = new AtomicInteger();
        Tokenizer tokenizer = new StopTokenizer();
        String output = TextGenerator.generate(ids -> endTokenLogits(calls),
                8, tokenizer, "p", 5, 1.0f, 1, 1L);

        assertEquals("p", output);
        assertEquals(1, calls.get());
    }

    @Test
    void outputGrowsWithMoreTokens() {
        String short_ = TextGenerator.generate(model, tok, cfg, "x", 2, 1.0f, 0, 5L);
        String long_  = TextGenerator.generate(model, tok, cfg, "x", 20, 1.0f, 0, 5L);

        assertTrue(long_.length() > short_.length(), "more tokens should produce longer output");
    }

    @Test
    void topKOneIsGreedy() {

        String a = TextGenerator.generate(model, tok, cfg, "hi", 10, 1.0f, 1, 1L);
        String b = TextGenerator.generate(model, tok, cfg, "hi", 10, 1.0f, 1, 999L);

        assertEquals(a, b, "topK=1 should be greedy and seed-independent");
    }

    @Test
    void topKZeroUsesFullVocab() {

        String out = TextGenerator.generate(model, tok, cfg, "hi", 5, 1.0f, 0, 1L);
        assertNotNull(out);
    }

    @Test
    void negativeMaxNewTokensThrows() {
        assertThrows(IllegalArgumentException.class,
                () -> TextGenerator.generate(model, tok, cfg, "x", -1, 1.0f, 0, 1L));
    }

    @Test
    void zeroTemperatureThrows() {
        assertThrows(IllegalArgumentException.class,
                () -> TextGenerator.generate(model, tok, cfg, "x", 5, 0.0f, 0, 1L));
    }

    @Test
    void negativeTemperatureThrows() {
        assertThrows(IllegalArgumentException.class,
                () -> TextGenerator.generate(model, tok, cfg, "x", 5, -0.5f, 0, 1L));
    }

    @Test
    void nonFiniteTemperatureThrows() {
        assertThrows(IllegalArgumentException.class,
                () -> TextGenerator.generate(model, tok, cfg, "x", 5, Float.NaN, 0, 1L));
    }

    @Test
    void emptyPromptThrowsWhenGenerating() {
        assertThrows(IllegalArgumentException.class,
                () -> TextGenerator.generate(model, tok, cfg, "", 1, 1.0f, 0, 1L));
    }

    @Test
    void negativeTopKThrows() {
        assertThrows(IllegalArgumentException.class,
                () -> TextGenerator.generate(model, tok, cfg, "x", 5, 1.0f, -1, 1L));
    }

    private static Tensor endTokenLogits(AtomicInteger calls) {
        calls.incrementAndGet();
        Tensor logits = new Tensor(1, 3);
        logits.data[0] = 0.0f;
        logits.data[1] = 10.0f;
        logits.data[2] = -1.0f;
        return logits;
    }

    private static final class StopTokenizer implements Tokenizer {

        @Override
        public int[] encode(String text) {
            return new int[]{0};
        }

        @Override
        public String decode(int[] ids) {
            return "p" + "x".repeat(Math.max(0, ids.length - 1));
        }

        @Override
        public int vocabSize() {
            return 3;
        }

        @Override
        public boolean isEndOfSequence(int tokenId) {
            return tokenId == 1;
        }
    }
}
