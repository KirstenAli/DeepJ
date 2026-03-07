package io.github.kirstenali.deepj.gpt;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.models.gpt.GPTConfig;
import io.github.kirstenali.deepj.models.gpt.GPTModel;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class GPTModelTest {

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
            dLogits.data[r][(r + 3) % dLogits.cols] = 1.0;
        }

        // zero all grads before backward
        for (Parameter p : model.parameters()) p.zeroGrad();

        model.backward(dLogits);

        // First parameter is token embedding weight (see GPTModel.parameters()).
        Parameter tokW = model.parameters().get(0);

        assertTrue(tokW.grad.sumAbs() > 0.0, "Expected non-zero token embedding grads");

        // Specific used ids should have non-zero rows.
        assertTrue (new Tensor(new double[][]{tokW.grad.data[5]}).sumAbs() > 0.0, "id=5 row grad should be non-zero");
        assertTrue(new Tensor(new double[][]{tokW.grad.data[1]}).sumAbs() > 0.0, "id=1 row grad should be non-zero");
        assertTrue(new Tensor(new double[][]{tokW.grad.data[2]}).sumAbs() > 0.0, "id=2 row grad should be non-zero");
    }
}
