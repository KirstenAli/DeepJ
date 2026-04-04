
package io.github.kirstenali.deepj.training;

import io.github.kirstenali.deepj.models.gpt.GPTConfig;
import io.github.kirstenali.deepj.models.gpt.GPTModel;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.data.TextDataset;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class CausalLMTrainingTest {

    @Test
    void trainer_runsOneStep_onTinyDataset() throws IOException {
        Tokenizer tok = new ByteTokenizer();
        Path tmp = Files.createTempFile("deepj_lm", ".txt");
        Files.writeString(tmp, "hello hello hello");

        TextDataset ds = TextDataset.fromFile(tmp, tok, 8, 1L);

        GPTConfig cfg = new GPTConfig(ByteTokenizer.VOCAB_SIZE, 8, 32, 4, 1, 64);
        GPTModel model = new GPTModel(cfg, 2L);

        Trainer trainer = CausalLMTraining.trainer(model, ds, 1e-2);

        double loss = trainer.trainStep(2);
        Assertions.assertTrue(Double.isFinite(loss));
        Assertions.assertTrue(loss > 0.0);
    }

    @Test
    void trainer_usesConfigGradClipNorm_toLimitUpdateMagnitude() throws IOException {
        Tokenizer tok = new ByteTokenizer();
        Path tmp = Files.createTempFile("deepj_lm_clip", ".txt");
        Files.writeString(tmp, "hello hello hello hello hello hello");

        TextDataset dsUnclipped = TextDataset.fromFile(tmp, tok, 8, 7L);
        TextDataset dsClipped = TextDataset.fromFile(tmp, tok, 8, 7L);

        GPTConfig cfgUnclipped = new GPTConfig(ByteTokenizer.VOCAB_SIZE, 8, 32, 4, 1, 64, 1.0, 1.0);
        GPTConfig cfgClipped = new GPTConfig(ByteTokenizer.VOCAB_SIZE, 8, 32, 4, 1, 64, 1.0, 1e-6);

        GPTModel modelUnclipped = new GPTModel(cfgUnclipped, 2L);
        GPTModel modelClipped = new GPTModel(cfgClipped, 2L);

        Trainer trainerUnclipped = CausalLMTraining.trainer(modelUnclipped, dsUnclipped, 1e-2);
        Trainer trainerClipped = CausalLMTraining.trainer(modelClipped, dsClipped, 1e-2);

        Parameter pUnclipped = modelUnclipped.parameters().get(0);
        Parameter pClipped = modelClipped.parameters().get(0);
        Tensor beforeUnclipped = pUnclipped.value.multiplyScalar(1.0);
        Tensor beforeClipped = pClipped.value.multiplyScalar(1.0);

        trainerUnclipped.trainStep(2);
        trainerClipped.trainStep(2);

        double deltaUnclipped = pUnclipped.value.subtract(beforeUnclipped).sumAbs();
        double deltaClipped = pClipped.value.subtract(beforeClipped).sumAbs();

        Assertions.assertTrue(deltaUnclipped > 0.0);
        Assertions.assertTrue(deltaClipped > 0.0);
        Assertions.assertTrue(deltaClipped < deltaUnclipped,
                "Expected clipped run to update less than unclipped run");
    }
}
