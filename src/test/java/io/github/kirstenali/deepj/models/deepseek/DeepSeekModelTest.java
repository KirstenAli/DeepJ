package io.github.kirstenali.deepj.models.deepseek;

import io.github.kirstenali.deepj.models.TextGenerator;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class DeepSeekModelTest {

    private DeepSeekConfig cfg;
    private DeepSeekModel model;

    @BeforeEach
    void setUp() {
        cfg = new DeepSeekConfig(
                ByteTokenizer.VOCAB_SIZE,
                16,   // maxSeqLen
                32,   // dModel
                4,    // nHeads
                2,    // nLayers
                64,   // dFF
                16,   // qRank
                8     // kvRank
        );
        model = new DeepSeekModel(cfg, 42L);
    }

    // ── config ─────────────────────────────────────────────────────

    @Test
    void config_rejectsInvalidCommonParams() {
        assertThrows(IllegalArgumentException.class,
                () -> new DeepSeekConfig(0, 16, 32, 4, 2, 64, 16, 8));
        assertThrows(IllegalArgumentException.class,
                () -> new DeepSeekConfig(256, 16, 33, 4, 2, 64, 16, 8)); // dModel % nHeads != 0
    }

    @Test
    void config_rejectsInvalidRanks() {
        assertThrows(IllegalArgumentException.class,
                () -> new DeepSeekConfig(256, 16, 32, 4, 2, 64, 0, 8));
        assertThrows(IllegalArgumentException.class,
                () -> new DeepSeekConfig(256, 16, 32, 4, 2, 64, 16, 0));
    }

    // ── forward ────────────────────────────────────────────────────

    @Test
    void forward_producesLogitsOfShape_seqLenByVocab() {
        int[] ids = {1, 2, 3, 4};
        Tensor logits = model.forward(ids);

        assertEquals(ids.length, logits.rows);
        assertEquals(cfg.vocabSize(), logits.cols);
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
        // tokEmb(1) + nLayers × DeepSeekBlock + normF(1) + lmHead(2: W+b)
        // DeepSeekBlock: ln1(1) + ln2(1) + MLA(6: Wdq,Wuq,Wdkv,Wuk,Wuv,Wo) + SwiGLU(3 Linear × 2 = 6) = 14
        int expectedPerBlock = 14;
        int expectedTotal = 1 + (cfg.nLayers() * expectedPerBlock) + 1 + 2;
        assertEquals(expectedTotal, model.parameters().size());
    }

    @Test
    void gradClipNorm_matchesConfig() {
        assertEquals(cfg.gradClipNorm(), model.gradClipNorm());
    }

    // ── generation ─────────────────────────────────────────────────

    @Test
    void generate_runsAndStartsWithPrompt() {
        Tokenizer tok = new ByteTokenizer();
        String out = TextGenerator.generate(model, tok, cfg, "hi", 8, 1.0, 0, 1L);

        assertNotNull(out);
        assertTrue(out.startsWith("hi"));
    }

    @Test
    void generate_sameSeedProducesSameOutput() {
        Tokenizer tok = new ByteTokenizer();
        String a = TextGenerator.generate(model, tok, cfg, "hello", 10, 0.8, 5, 42L);
        String b = TextGenerator.generate(model, tok, cfg, "hello", 10, 0.8, 5, 42L);

        assertEquals(a, b);
    }
}

