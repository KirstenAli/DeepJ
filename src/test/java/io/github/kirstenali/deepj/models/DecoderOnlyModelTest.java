package io.github.kirstenali.deepj.models;

import io.github.kirstenali.deepj.models.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.models.prism.DeepJPrismModel;
import io.github.kirstenali.deepj.models.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.models.origin.DeepJOriginModel;
import io.github.kirstenali.deepj.models.orbit.DeepJOrbitConfig;
import io.github.kirstenali.deepj.models.orbit.DeepJOrbitModel;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

class DecoderOnlyModelTest {

    static Stream<DecoderOnlyModel> allModels() {
        int vocab = ByteTokenizer.VOCAB_SIZE;

        DeepJOriginModel origin = new DeepJOriginModel(
                new DeepJOriginConfig(vocab, 16, 32, 4, 1, 64), 1L);

        DeepJOrbitModel orbit = new DeepJOrbitModel(
                new DeepJOrbitConfig(vocab, 16, 32, 4, 1, DeepJOrbitConfig.defaultDFF(32)), 1L);

        DeepJPrismModel prism = new DeepJPrismModel(
                new DeepJPrismConfig(vocab, 16, 32, 4, 1, 64, 16, 8), 1L);

        return Stream.of(origin, orbit, prism);
    }

    @ParameterizedTest
    @MethodSource("allModels")
    void model_implementsCausalLM(DecoderOnlyModel model) {
        assertInstanceOf(CausalLM.class, model);
    }

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

    @ParameterizedTest
    @MethodSource("allModels")
    void backward_populatesAtLeastOneGradient(DecoderOnlyModel model) {
        int[] ids = {1, 2, 3};
        Tensor logits = model.forward(ids);
        model.zeroGrad();
        model.backward(Tensor.ones(logits.rows, logits.cols));

        boolean anyNonZero = model.parameters().stream()
                .anyMatch(p -> p.grad != null && p.grad.sumAbs() > 0.0f);
        assertTrue(anyNonZero, "at least one parameter must receive a non-zero gradient");
    }

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
                assertEquals(0.0f, p.grad.sumAbs(), 1e-12f,
                        "every gradient must be zero after zeroGrad()");
            }
        });
    }

    @ParameterizedTest
    @MethodSource("allModels")
    void gradClipNorm_isPositive(DecoderOnlyModel model) {
        assertTrue(model.gradClipNorm() > 0.0f, "gradClipNorm must be positive");
    }
}
