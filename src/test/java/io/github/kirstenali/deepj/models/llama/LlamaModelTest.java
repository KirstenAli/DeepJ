package io.github.kirstenali.deepj.models.llama;

import io.github.kirstenali.deepj.models.TextGenerator;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

public class LlamaModelTest {

    @TempDir
    Path temporaryDirectory;

    private LlamaConfig cfg;
    private LlamaModel model;

    @BeforeEach
    void setUp() {
        cfg = new LlamaConfig(
                ByteTokenizer.VOCAB_SIZE,
                16,
                32,
                4,
                2,
                LlamaConfig.defaultDFF(32)
        );
        model = new LlamaModel(cfg, 42L);
    }

    @Test
    void config_defaultDFF_isRoundedMultipleOf64() {
        int dFF = LlamaConfig.defaultDFF(32);
        assertEquals(0, dFF % 64, "defaultDFF must be a multiple of 64");
        assertTrue(dFF > 0, "defaultDFF must be positive");
    }

    @Test
    void configIncludesStableTrainingDefaults() {
        assertEquals(0.2f, cfg.initScale(), 1e-12f);
        assertEquals(1.0f, cfg.gradClipNorm(), 1e-12f);
        assertThrows(IllegalArgumentException.class,
                () -> new LlamaConfig(11, 8, 4, 2, 1, 8, 0.0f, 1.0f));
    }

    @Test
    void modelAppliesInitScaleAndUsesIndependentStreams() {
        LlamaConfig base = new LlamaConfig(11, 8, 4, 2, 1, 8, 1.0f, 1.0f);
        LlamaConfig scaled = new LlamaConfig(11, 8, 4, 2, 1, 8, 0.2f, 1.0f);
        LlamaModel baseModel = new LlamaModel(base, 1234L);
        LlamaModel scaledModel = new LlamaModel(scaled, 1234L);
        float ratio = scaledModel.parameters().get(0).value.sumAbs()
                / baseModel.parameters().get(0).value.sumAbs();
        assertEquals(0.2f, ratio, 1e-6f);
        assertTrue(scaledModel.parameters().stream().anyMatch(p -> isAllOnes(p.value)));
        assertNotEquals(scaledModel.parameters().get(0).value.data[0],
                scaledModel.parameters().get(3).value.data[0]);
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

        assertThrows(IllegalArgumentException.class,
                () -> new LlamaConfig(256, 16, 33, 4, 2, 64));

        assertThrows(IllegalArgumentException.class,
                () -> new LlamaConfig(256, 16, 6, 2, 2, 64));

        assertThrows(IllegalArgumentException.class,
                () -> new LlamaConfig(256, 16, 32, 4, 2, 64, 0.0f));
    }

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

    @Test
    void backward_accumulatesGradients() {
        int[] ids = {1, 2, 3};
        Tensor logits = model.forward(ids);
        Tensor dLogits = Tensor.ones(logits.rows, logits.cols);

        model.parameters().forEach(p -> p.zeroGrad());
        model.backward(dLogits);

        boolean anyNonZero = model.parameters().stream()
                .anyMatch(p -> p.grad.sumAbs() > 0.0f);
        assertTrue(anyNonZero, "at least one parameter gradient must be non-zero after backward");
    }

    @Test
    void parameters_countMatchesExpectedStructure() {

        int expectedPerBlock = 12;
        int expectedTotal = 1 + (cfg.nLayers() * expectedPerBlock) + 1 + 2;
        assertEquals(expectedTotal, model.parameters().size());
    }

    @Test
    void gradClipNorm_matchesConfig() {
        assertEquals(cfg.gradClipNorm(), model.gradClipNorm());
        assertSame(cfg, model.config());
    }

    @Test
    void checkpointRoundTripPreservesLogits() throws IOException {
        int[] ids = {1, 2, 3};
        float[] expected = materializedData(model.forward(ids));
        Path checkpoint = temporaryDirectory.resolve("llama.dj");
        model.save(checkpoint);
        LlamaModel restored = new LlamaModel(cfg, 99L);
        restored.load(checkpoint);
        assertArrayEquals(expected, materializedData(restored.forward(ids)));
    }

    @Test
    void generate_runsAndReturnsNonEmptyString() {
        Tokenizer tok = new ByteTokenizer();
        String out = TextGenerator.generate(model, tok, cfg, "hi", 8, 1.0f, 0, 1L);

        assertNotNull(out);
        assertTrue(out.startsWith("hi"), "output must begin with the prompt");
    }

    @Test
    void generate_sameSeedProducesSameOutput() {
        Tokenizer tok = new ByteTokenizer();
        String a = TextGenerator.generate(model, tok, cfg, "hello", 10, 0.8f, 5, 42L);
        String b = TextGenerator.generate(model, tok, cfg, "hello", 10, 0.8f, 5, 42L);

        assertEquals(a, b, "identical seeds must produce identical output");
    }

    private static float[] materializedData(Tensor tensor) {
        tensor.materialize();
        return tensor.data.clone();
    }

    private static boolean isAllOnes(Tensor tensor) {
        for (float value : tensor.data) if (value != 1.0f) return false;
        return true;
    }
}
