
package io.github.kirstenali.deepj.training;

import io.github.kirstenali.deepj.models.gpt.GPTConfig;
import io.github.kirstenali.deepj.models.gpt.GPTModel;
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
}
