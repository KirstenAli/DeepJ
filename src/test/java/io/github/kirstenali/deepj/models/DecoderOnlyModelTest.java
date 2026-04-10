package io.github.kirstenali.deepj.models;

import io.github.kirstenali.deepj.models.deepseek.DeepSeekConfig;
import io.github.kirstenali.deepj.models.deepseek.DeepSeekModel;
import io.github.kirstenali.deepj.models.gpt.GPTConfig;
import io.github.kirstenali.deepj.models.gpt.GPTModel;
import io.github.kirstenali.deepj.models.llama.LlamaConfig;
import io.github.kirstenali.deepj.models.llama.LlamaModel;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies the shared contract defined by {@link DecoderOnlyModel} across all three
 * concrete subclasses: GPTModel, LlamaModel, and DeepSeekModel.
 */
class DecoderOnlyModelTest {

    // ── Fixtures ───────────────────────────────────────────────────

    static Stream<DecoderOnlyModel> allModels() {
        int vocab = ByteTokenizer.VOCAB_SIZE;

        GPTModel gpt = new GPTModel(
                new GPTConfig(vocab, 16, 32, 4, 1, 64), 1L);

        LlamaModel llama = new LlamaModel(
                new LlamaConfig(vocab, 16, 32, 4, 1, LlamaConfig.defaultDFF(32)), 1L);

        DeepSeekModel deepSeek = new DeepSeekModel(
                new DeepSeekConfig(vocab, 16, 32, 4, 1, 64, 16, 8), 1L);

        return Stream.of(gpt, llama, deepSeek);
    }

    // ── Type hierarchy ─────────────────────────────────────────────

    @ParameterizedTest
    @MethodSource("allModels")
    void model_implementsCausalLM(DecoderOnlyModel model) {
        assertInstanceOf(CausalLM.class, model);
    }

    // ── Forward ────────────────────────────────────────────────────

    @ParameterizedTest
    @MethodSource("allModels")
    void forward_returnsCorrectShape(DecoderOnlyModel model) {
        int[] ids = {1, 2, 3, 4};
        Tensor logits = model.forward(ids);

        assertNotNull(logits);
        assertEquals(ids.length, logits.rows, "logits rows must equal seqLen");
        assertEquals(ByteTokenizer.VOCAB_SIZE, logits.cols, "logits cols must equal vocabSize");
    }

    @ParameterizedTest
    @MethodSource("allModels")
    void forward_singleToken_doesNotThrow(DecoderOnlyModel model) {
        assertDoesNotThrow(() -> model.forward(new int[]{0}));
    }

    @ParameterizedTest
    @MethodSource("allModels")
    void forward_allFiniteValues(DecoderOnlyModel model) {
        Tensor logits = model.forward(new int[]{1, 2, 3});
        for (int r = 0; r < logits.rows; r++) {
            for (int c = 0; c < logits.cols; c++) {
                assertTrue(Double.isFinite(logits.data[r * logits.cols + c]),
                        "logit[" + r + "][" + c + "] must be finite");
            }
        }
    }

    // ── Backward ───────────────────────────────────────────────────

    @ParameterizedTest
    @MethodSource("allModels")
    void backward_populatesAtLeastOneGradient(DecoderOnlyModel model) {
        int[] ids = {1, 2, 3};
        Tensor logits = model.forward(ids);
        model.zeroGrad();
        model.backward(Tensor.ones(logits.rows, logits.cols));

        boolean anyNonZero = model.parameters().stream()
                .anyMatch(p -> p.grad != null && p.grad.sumAbs() > 0.0);
        assertTrue(anyNonZero, "at least one parameter must receive a non-zero gradient");
    }

    // ── Parameters ─────────────────────────────────────────────────

    @ParameterizedTest
    @MethodSource("allModels")
    void parameters_nonEmptyAndNoBias(DecoderOnlyModel model) {
        assertFalse(model.parameters().isEmpty(), "parameter list must not be empty");
    }

    @ParameterizedTest
    @MethodSource("allModels")
    void zeroGrad_clearsAllGradients(DecoderOnlyModel model) {
        int[] ids = {1, 2};
        Tensor logits = model.forward(ids);
        model.backward(Tensor.ones(logits.rows, logits.cols));

        model.zeroGrad();

        model.parameters().forEach(p -> {
            if (p.grad != null) {
                assertEquals(0.0, p.grad.sumAbs(), 1e-12,
                        "every gradient must be zero after zeroGrad()");
            }
        });
    }

    // ── gradClipNorm ───────────────────────────────────────────────

    @ParameterizedTest
    @MethodSource("allModels")
    void gradClipNorm_isPositive(DecoderOnlyModel model) {
        assertTrue(model.gradClipNorm() > 0.0, "gradClipNorm must be positive");
    }
}

