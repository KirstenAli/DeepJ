package io.github.kirstenali.deepj.training;

import io.github.kirstenali.deepj.data.TextDataset;
import io.github.kirstenali.deepj.models.CausalLM;
import io.github.kirstenali.deepj.models.deepseek.DeepSeekConfig;
import io.github.kirstenali.deepj.models.deepseek.DeepSeekModel;
import io.github.kirstenali.deepj.models.gpt.GPTConfig;
import io.github.kirstenali.deepj.models.gpt.GPTModel;
import io.github.kirstenali.deepj.models.llama.LlamaConfig;
import io.github.kirstenali.deepj.models.llama.LlamaModel;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;

public class CausalLMTrainingTest {

    // ── Helpers ───────────────────────────────────────────────────

    private static TextDataset tinyDataset(String text, int seqLen) throws IOException {
        Path tmp = Files.createTempFile("deepj_lm", ".txt");
        Files.writeString(tmp, text);
        return TextDataset.fromFile(tmp, new ByteTokenizer(), seqLen, 1L);
    }

    static Stream<CausalLM> allModels() {
        int vocab = ByteTokenizer.VOCAB_SIZE;
        return Stream.of(
                new GPTModel(new GPTConfig(vocab, 8, 32, 4, 1, 64), 1L),
                new LlamaModel(new LlamaConfig(vocab, 8, 32, 4, 1, LlamaConfig.defaultDFF(32)), 1L),
                new DeepSeekModel(new DeepSeekConfig(vocab, 8, 32, 4, 1, 64, 16, 8), 1L)
        );
    }

    // ── CausalLM interface ────────────────────────────────────────

    @ParameterizedTest
    @MethodSource("allModels")
    void allModels_implementCausalLM(CausalLM model) {
        Assertions.assertNotNull(model);
    }

    // ── Generic trainer ───────────────────────────────────────────

    @ParameterizedTest
    @MethodSource("allModels")
    void trainer_runsOneStep_onAllModelTypes(CausalLM model) throws IOException {
        TextDataset ds = tinyDataset("hello hello hello hello hello", 8);
        Trainer trainer = CausalLMTraining.trainer(model, ds, 1e-2f);

        double loss = trainer.trainStep(2);

        Assertions.assertTrue(Double.isFinite(loss), "loss must be finite");
        Assertions.assertTrue(loss > 0.0f, "loss must be positive");
    }

    @ParameterizedTest
    @MethodSource("allModels")
    void trainer_updatesParameters_afterOneStep(CausalLM model) throws IOException {
        TextDataset ds = tinyDataset("hello hello hello hello hello", 8);

        Tensor before = model.parameters().get(0).value.multiplyScalar(1.0f);

        Trainer trainer = CausalLMTraining.trainer(model, ds, 1e-2f);
        trainer.trainStep(2);

        double delta = model.parameters().get(0).value.subtract(before).sumAbs();
        Assertions.assertTrue(delta > 0.0f, "parameters must change after a training step");
    }

    // ── GPT-specific tests ────────────────────────────────────────

    @Test
    void trainer_runsOneStep_onTinyDataset() throws IOException {
        Tokenizer tok = new ByteTokenizer();
        Path tmp = Files.createTempFile("deepj_lm", ".txt");
        Files.writeString(tmp, "hello hello hello");

        TextDataset ds = TextDataset.fromFile(tmp, tok, 8, 1L);

        GPTConfig cfg = new GPTConfig(ByteTokenizer.VOCAB_SIZE, 8, 32, 4, 1, 64);
        GPTModel model = new GPTModel(cfg, 2L);

        Trainer trainer = CausalLMTraining.trainer(model, ds, 1e-2f);

        double loss = trainer.trainStep(2);
        Assertions.assertTrue(Double.isFinite(loss));
        Assertions.assertTrue(loss > 0.0f);
    }

    @Test
    void trainer_usesConfigGradClipNorm_toLimitUpdateMagnitude() throws IOException {
        Tokenizer tok = new ByteTokenizer();
        Path tmp = Files.createTempFile("deepj_lm_clip", ".txt");
        Files.writeString(tmp, "hello hello hello hello hello hello");

        TextDataset dsUnclipped = TextDataset.fromFile(tmp, tok, 8, 7L);
        TextDataset dsClipped   = TextDataset.fromFile(tmp, tok, 8, 7L);

        GPTConfig cfgUnclipped = new GPTConfig(ByteTokenizer.VOCAB_SIZE, 8, 32, 4, 1, 64, 1.0f, 1.0f);
        GPTConfig cfgClipped   = new GPTConfig(ByteTokenizer.VOCAB_SIZE, 8, 32, 4, 1, 64, 1.0f, 1e-6f);

        GPTModel modelUnclipped = new GPTModel(cfgUnclipped, 2L);
        GPTModel modelClipped   = new GPTModel(cfgClipped, 2L);

        Trainer trainerUnclipped = CausalLMTraining.trainer(modelUnclipped, dsUnclipped, 1e-2f);
        Trainer trainerClipped   = CausalLMTraining.trainer(modelClipped,   dsClipped,   1e-2f);

        Parameter pUnclipped = modelUnclipped.parameters().get(0);
        Parameter pClipped   = modelClipped.parameters().get(0);
        Tensor beforeUnclipped = pUnclipped.value.multiplyScalar(1.0f);
        Tensor beforeClipped   = pClipped.value.multiplyScalar(1.0f);

        trainerUnclipped.trainStep(2);
        trainerClipped.trainStep(2);

        double deltaUnclipped = pUnclipped.value.subtract(beforeUnclipped).sumAbs();
        double deltaClipped   = pClipped.value.subtract(beforeClipped).sumAbs();

        Assertions.assertTrue(deltaUnclipped > 0.0f);
        Assertions.assertTrue(deltaClipped > 0.0f);
        Assertions.assertTrue(deltaClipped < deltaUnclipped,
                "Expected clipped run to update less than unclipped run");
    }
}

