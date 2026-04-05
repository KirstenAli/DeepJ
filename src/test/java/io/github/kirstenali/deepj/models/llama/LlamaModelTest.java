package io.github.kirstenali.deepj.models.llama;

import io.github.kirstenali.deepj.models.TextGenerator;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class LlamaModelTest {

    private LlamaConfig cfg;
    private LlamaModel model;

    @BeforeEach
    void setUp() {
        cfg = new LlamaConfig(
                ByteTokenizer.VOCAB_SIZE,
                16,   // maxSeqLen
                32,   // dModel
                4,    // nHeads
                2,    // nLayers
                LlamaConfig.defaultDFF(32)
        );
        model = new LlamaModel(cfg, 42L);
    }

    // ── config ─────────────────────────────────────────────────────

    @Test
    void config_defaultDFF_isRoundedMultipleOf64() {
        int dFF = LlamaConfig.defaultDFF(32);
        assertEquals(0, dFF % 64, "defaultDFF must be a multiple of 64");
        assertTrue(dFF > 0, "defaultDFF must be positive");
    }

    @Test
    void config_rejectsInvalidParams() {
        assertThrows(IllegalArgumentException.class,
                () -> new LlamaConfig(0, 16, 32, 4, 2, 64));
        assertThrows(IllegalArgumentException.class,
                () -> new LlamaConfig(256, 0, 32, 4, 2, 64));
        assertThrows(IllegalArgumentException.class,
                () -> new LlamaConfig(256, 16, 0, 4, 2, 64));
        assertThrows(IllegalArgumentException.class,
                () -> new LlamaConfig(256, 16, 32, 0, 2, 64));
        assertThrows(IllegalArgumentException.class,
                () -> new LlamaConfig(256, 16, 32, 4, 0, 64));
        assertThrows(IllegalArgumentException.class,
                () -> new LlamaConfig(256, 16, 32, 4, 2, 0));
        // dModel not divisible by nHeads
        assertThrows(IllegalArgumentException.class,
                () -> new LlamaConfig(256, 16, 33, 4, 2, 64));
        // invalid gradClipNorm
        assertThrows(IllegalArgumentException.class,
                () -> new LlamaConfig(256, 16, 32, 4, 2, 64, 0.0));
    }

    // ── forward ────────────────────────────────────────────────────

    @Test
    void forward_producesLogitsOfShape_seqLenByVocab() {
        int[] ids = {1, 2, 3, 4};
        Tensor logits = model.forward(ids);

        assertEquals(ids.length, logits.rows, "logits rows must equal seqLen");
        assertEquals(cfg.vocabSize(), logits.cols, "logits cols must equal vocabSize");
    }

    @Test
    void forward_singleToken_doesNotThrow() {
        Tensor logits = model.forward(new int[]{5});
        assertEquals(1, logits.rows);
        assertEquals(cfg.vocabSize(), logits.cols);
    }

    @Test
    void forward_fullContextWindow_doesNotThrow() {
        int[] ids = new int[cfg.maxSeqLen()];
        assertDoesNotThrow(() -> model.forward(ids));
    }

    // ── backward ───────────────────────────────────────────────────

    @Test
    void backward_accumulatesGradients() {
        int[] ids = {1, 2, 3};
        Tensor logits = model.forward(ids);
        Tensor dLogits = Tensor.ones(logits.rows, logits.cols);

        model.parameters().forEach(p -> p.zeroGrad());
        model.backward(dLogits);

        boolean anyNonZero = model.parameters().stream()
                .anyMatch(p -> p.grad.sumAbs() > 0.0);
        assertTrue(anyNonZero, "at least one parameter gradient must be non-zero after backward");
    }

    // ── parameters ─────────────────────────────────────────────────

    @Test
    void parameters_countMatchesExpectedStructure() {
        // tokEmb(1) + nLayers × LlamaBlock + normF(1) + lmHead(2: W+b)
        // LlamaBlock: ln1(1) + ln2(1) + attn(4:Wq,Wk,Wv,Wo) + SwiGLU(3 Linear × 2:W+b = 6) = 12
        int expectedPerBlock = 12;
        int expectedTotal = 1 + (cfg.nLayers() * expectedPerBlock) + 1 + 2;
        assertEquals(expectedTotal, model.parameters().size());
    }

    @Test
    void gradClipNorm_matchesConfig() {
        assertEquals(cfg.gradClipNorm(), model.gradClipNorm());
    }

    // ── generation ─────────────────────────────────────────────────

    @Test
    void generate_runsAndReturnsNonEmptyString() {
        Tokenizer tok = new ByteTokenizer();
        String out = TextGenerator.generate(model, tok, cfg, "hi", 8, 1.0, 0, 1L);

        assertNotNull(out);
        assertTrue(out.startsWith("hi"), "output must begin with the prompt");
    }

    @Test
    void generate_sameSeedProducesSameOutput() {
        Tokenizer tok = new ByteTokenizer();
        String a = TextGenerator.generate(model, tok, cfg, "hello", 10, 0.8, 5, 42L);
        String b = TextGenerator.generate(model, tok, cfg, "hello", 10, 0.8, 5, 42L);

        assertEquals(a, b, "identical seeds must produce identical output");
    }
}

