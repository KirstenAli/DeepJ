package io.github.kirstenali.deepj.models.origin;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

public class DeepJOriginTest {

    @TempDir
    Path temporaryDirectory;

    @Test
    void config_defaultsIncludeStabilityKnobs() {
        DeepJOriginConfig cfg = new DeepJOriginConfig(11, 8, 4, 2, 1, 8);
        assertEquals(0.2f, cfg.initScale(), 1e-12f);
        assertEquals(1.0f, cfg.gradClipNorm(), 1e-12f);
    }

    @Test
    void model_appliesInitScaleFromConfig() {
        DeepJOriginConfig base = new DeepJOriginConfig(11, 8, 4, 2, 1, 8, 1.0f, 1.0f);
        DeepJOriginConfig scaled = new DeepJOriginConfig(11, 8, 4, 2, 1, 8, 0.2f, 1.0f);

        DeepJOrigin mBase = new DeepJOrigin(base, 1234L);
        DeepJOrigin mScaled = new DeepJOrigin(scaled, 1234L);

        double baseAbs = mBase.parameters().get(0).value.sumAbs();
        double scaledAbs = mScaled.parameters().get(0).value.sumAbs();

        assertTrue(baseAbs > 0.0f);
        assertEquals(0.2f, scaledAbs / baseAbs, 1e-6f);
        assertLayerNormGainsRemainOne(mScaled);
        assertEquals(1.0f, mBase.gradClipNorm(), 1e-12f);
    }

    @Test
    void modelUsesIndependentInitializationStreams() {
        DeepJOrigin model = new DeepJOrigin(new DeepJOriginConfig(11, 8, 4, 2, 1, 8), 1234L);
        float embeddingFirst = model.parameters().get(0).value.data[0];
        float attentionFirst = model.parameters().get(6).value.data[0];
        assertNotEquals(embeddingFirst, attentionFirst);
    }

    private static void assertLayerNormGainsRemainOne(DeepJOrigin model) {
        long unitParameters = model.parameters().stream()
                .map(parameter -> parameter.value)
                .filter(DeepJOriginTest::isAllOnes)
                .count();
        assertTrue(unitParameters > 0, "LayerNorm gains must not be scaled");
    }

    private static boolean isAllOnes(Tensor tensor) {
        for (float value : tensor.data) {
            if (value != 1.0f) return false;
        }
        return true;
    }

    @Test
    void config_rejectsInvalidStabilityKnobs() {
        assertThrows(IllegalArgumentException.class,
                () -> new DeepJOriginConfig(11, 8, 4, 2, 1, 8, 0.0f, 1.0f));
        assertThrows(IllegalArgumentException.class,
                () -> new DeepJOriginConfig(11, 8, 4, 2, 1, 8, Float.NaN, 1.0f));
        assertThrows(IllegalArgumentException.class,
                () -> new DeepJOriginConfig(11, 8, 4, 2, 1, 8, 1.0f, 0.0f));
        assertThrows(IllegalArgumentException.class,
                () -> new DeepJOriginConfig(11, 8, 4, 2, 1, 8, 1.0f, Float.NaN));
    }

    @Test
    void config_rejectsInvalidCoreParams() {
        assertThrows(IllegalArgumentException.class,
                () -> new DeepJOriginConfig(0, 8, 4, 2, 1, 8));
        assertThrows(IllegalArgumentException.class,
                () -> new DeepJOriginConfig(11, 0, 4, 2, 1, 8));
        assertThrows(IllegalArgumentException.class,
                () -> new DeepJOriginConfig(11, 8, 0, 2, 1, 8));
        assertThrows(IllegalArgumentException.class,
                () -> new DeepJOriginConfig(11, 8, 4, 0, 1, 8));
        assertThrows(IllegalArgumentException.class,
                () -> new DeepJOriginConfig(11, 8, 4, 2, 0, 8));
        assertThrows(IllegalArgumentException.class,
                () -> new DeepJOriginConfig(11, 8, 4, 2, 1, 0));
        assertThrows(IllegalArgumentException.class,
                () -> new DeepJOriginConfig(11, 8, 5, 2, 1, 8));
    }

    @Test
    void forward_producesLogitsOfShape_seqLenByVocab() {
        DeepJOriginConfig cfg = new DeepJOriginConfig(
                11,
                8,
                4,
                2,
                1,
                8
        );

        DeepJOrigin model = new DeepJOrigin(cfg, 1234L);

        int[] ids = new int[]{1, 2, 3, 4};
        Tensor logits = model.forward(ids);

        TestSupport.assertTensorShape(logits, ids.length, cfg.vocabSize());
    }

    @Test
    void parameters_countMatchesExpected_forSmallConfig() {
        DeepJOriginConfig cfg = new DeepJOriginConfig(10, 8, 4, 2, 1, 8);
        DeepJOrigin model = new DeepJOrigin(cfg, 1L);

        int expected = 1 + 1 + 12 * cfg.nLayers() + 2 + 2;
        assertEquals(expected, model.parameters().size());
    }

    @Test
    void backward_setsNonZeroGrads_forTokenEmbeddingRowsUsed() {
        DeepJOriginConfig cfg = new DeepJOriginConfig(13, 8, 4, 2, 1, 8);
        DeepJOrigin model = new DeepJOrigin(cfg, 99L);

        int[] ids = new int[]{5, 1, 5, 2};
        Tensor logits = model.forward(ids);

        Tensor dLogits = Tensor.zeros(logits.rows, logits.cols);
        for (int r = 0; r < dLogits.rows; r++) {
            dLogits.data[r * dLogits.cols + (r + 3) % dLogits.cols] = 1.0f;
        }

        for (Parameter p : model.parameters()) {
            p.zeroGrad();
        }

        model.backward(dLogits);

        Parameter tokW = model.parameters().get(0);

        assertTrue(tokW.grad.sumAbs() > 0.0f, "Expected non-zero token embedding grads");

        assertTrue (tokW.grad.getRow(5).sumAbs() > 0.0f, "id=5 row grad should be non-zero");
        assertTrue(tokW.grad.getRow(1).sumAbs() > 0.0f, "id=1 row grad should be non-zero");
        assertTrue(tokW.grad.getRow(2).sumAbs() > 0.0f, "id=2 row grad should be non-zero");
    }

    @Test
    void checkpointRoundTripPreservesLogits() throws IOException {
        DeepJOriginConfig config = new DeepJOriginConfig(11, 8, 4, 2, 1, 8);
        DeepJOrigin original = new DeepJOrigin(config, 1L);
        int[] ids = {1, 2, 3};
        float[] expected = materializedData(original.forward(ids));
        Path checkpoint = temporaryDirectory.resolve("origin.dj");
        original.save(checkpoint);
        DeepJOrigin restored = new DeepJOrigin(config, 2L);
        restored.load(checkpoint);
        assertArrayEquals(expected, materializedData(restored.forward(ids)));
    }

    private static float[] materializedData(Tensor tensor) {
        tensor.materialize();
        return tensor.data.clone();
    }
}
