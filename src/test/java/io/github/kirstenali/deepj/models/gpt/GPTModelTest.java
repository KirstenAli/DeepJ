package io.github.kirstenali.deepj.models.gpt;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class GPTModelTest {

    @Test
    void config_defaultsIncludeStabilityKnobs() {
        GPTConfig cfg = new GPTConfig(11, 8, 4, 2, 1, 8);
        assertEquals(0.2f, cfg.initScale(), 1e-12f);
        assertEquals(1.0f, cfg.gradClipNorm(), 1e-12f);
    }

    @Test
    void model_appliesInitScaleFromConfig() {
        GPTConfig base = new GPTConfig(11, 8, 4, 2, 1, 8, 1.0f, 1.0f);
        GPTConfig scaled = new GPTConfig(11, 8, 4, 2, 1, 8, 0.2f, 1.0f);

        GPTModel mBase = new GPTModel(base, 1234L);
        GPTModel mScaled = new GPTModel(scaled, 1234L);

        double baseAbs = mBase.parameters().get(0).value.sumAbs();
        double scaledAbs = mScaled.parameters().get(0).value.sumAbs();

        assertTrue(baseAbs > 0.0f);
        assertEquals(0.2f, scaledAbs / baseAbs, 1e-6f);
        assertEquals(1.0f, mBase.gradClipNorm(), 1e-12f);
    }

    @Test
    void config_rejectsInvalidStabilityKnobs() {
        assertThrows(IllegalArgumentException.class,
                () -> new GPTConfig(11, 8, 4, 2, 1, 8, 0.0f, 1.0f));
        assertThrows(IllegalArgumentException.class,
                () -> new GPTConfig(11, 8, 4, 2, 1, 8, Float.NaN, 1.0f));
        assertThrows(IllegalArgumentException.class,
                () -> new GPTConfig(11, 8, 4, 2, 1, 8, 1.0f, 0.0f));
        assertThrows(IllegalArgumentException.class,
                () -> new GPTConfig(11, 8, 4, 2, 1, 8, 1.0f, Float.NaN));
    }

    @Test
    void config_rejectsInvalidCoreParams() {
        assertThrows(IllegalArgumentException.class,
                () -> new GPTConfig(0, 8, 4, 2, 1, 8));
        assertThrows(IllegalArgumentException.class,
                () -> new GPTConfig(11, 0, 4, 2, 1, 8));
        assertThrows(IllegalArgumentException.class,
                () -> new GPTConfig(11, 8, 0, 2, 1, 8));
        assertThrows(IllegalArgumentException.class,
                () -> new GPTConfig(11, 8, 4, 0, 1, 8));
        assertThrows(IllegalArgumentException.class,
                () -> new GPTConfig(11, 8, 4, 2, 0, 8));
        assertThrows(IllegalArgumentException.class,
                () -> new GPTConfig(11, 8, 4, 2, 1, 0));
        assertThrows(IllegalArgumentException.class,
                () -> new GPTConfig(11, 8, 5, 2, 1, 8));
    }

    @Test
    void forward_producesLogitsOfShape_seqLenByVocab() {
        GPTConfig cfg = new GPTConfig(
                11, // vocab
                8,  // max seq
                4,  // dModel
                2,  // heads
                1,  // layers
                8   // dFF
        );

        GPTModel model = new GPTModel(cfg, 1234L);

        int[] ids = new int[]{1, 2, 3, 4};
        Tensor logits = model.forward(ids);

        TestSupport.assertTensorShape(logits, ids.length, cfg.vocabSize());
    }

    @Test
    void parameters_countMatchesExpected_forSmallConfig() {
        GPTConfig cfg = new GPTConfig(10, 8, 4, 2, 1, 8);
        GPTModel model = new GPTModel(cfg, 1L);

        // Expected:
        // tokEmb: 1
        // posEmb: 1
        // per block: ln1(2) + ln2(2) + attn(4) + mlp (2 Linear => 4 params) = 12
        // lnF: 2
        // lmHead: 2
        int expected = 1 + 1 + 12 * cfg.nLayers() + 2 + 2;
        assertEquals(expected, model.parameters().size());
    }

    @Test
    void backward_setsNonZeroGrads_forTokenEmbeddingRowsUsed() {
        GPTConfig cfg = new GPTConfig(13, 8, 4, 2, 1, 8);
        GPTModel model = new GPTModel(cfg, 99L);

        int[] ids = new int[]{5, 1, 5, 2};
        Tensor logits = model.forward(ids);

        // upstream grad: make it non-uniform to reduce risk of accidental cancellation
        Tensor dLogits = Tensor.zeros(logits.rows, logits.cols);
        for (int r = 0; r < dLogits.rows; r++) {
            dLogits.data[r * dLogits.cols + (r + 3) % dLogits.cols] = 1.0f;
        }

        // zero all grads before backward
        for (Parameter p : model.parameters()) p.zeroGrad();

        model.backward(dLogits);

        // First parameter is token embedding weight (see GPTModel.parameters()).
        Parameter tokW = model.parameters().get(0);

        assertTrue(tokW.grad.sumAbs() > 0.0f, "Expected non-zero token embedding grads");

        // Specific used ids should have non-zero rows.
        assertTrue (tokW.grad.getRow(5).sumAbs() > 0.0f, "id=5 row grad should be non-zero");
        assertTrue(tokW.grad.getRow(1).sumAbs() > 0.0f, "id=1 row grad should be non-zero");
        assertTrue(tokW.grad.getRow(2).sumAbs() > 0.0f, "id=2 row grad should be non-zero");
    }
}
