package io.github.kirstenali.deepj.training;

import io.github.kirstenali.deepj.data.TextDataset;
import io.github.kirstenali.deepj.models.CausalLM;
import io.github.kirstenali.deepj.models.deepseek.DeepSeekConfig;
import io.github.kirstenali.deepj.models.deepseek.DeepSeekModel;
import io.github.kirstenali.deepj.models.gpt.GPTConfig;
import io.github.kirstenali.deepj.models.gpt.GPTModel;
import io.github.kirstenali.deepj.models.llama.LlamaConfig;
import io.github.kirstenali.deepj.models.llama.LlamaModel;
import io.github.kirstenali.deepj.optimisers.AdamW;
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
import java.util.List;
import java.util.stream.Stream;

public class CausalLMTrainingTest {

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

    @ParameterizedTest
    @MethodSource("allModels")
    void allModels_implementCausalLM(CausalLM model) {
        Assertions.assertNotNull(model);
    }

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
    void accumulationMatchesEquivalentBatch() throws IOException {
        GPTModel accumulated = gptModel(3L);
        GPTModel batched = gptModel(3L);
        float accumulatedLoss = accumulatedLoss(accumulated);
        float batchedLoss = batchedLoss(batched);

        Assertions.assertEquals(batchedLoss, accumulatedLoss, 1e-6f);
        assertParametersClose(batched.parameters(), accumulated.parameters());
    }

    @Test
    void accumulationStepsMustBePositive() throws IOException {
        GPTModel model = gptModel(3L);
        TextDataset dataset = accumulationDataset();
        var optimizer = AdamW.defaultAdamW(1e-2f);

        Assertions.assertThrows(IllegalArgumentException.class,
                () -> CausalLMTraining.trainer(model, dataset, optimizer, 0));
    }

    @Test
    void trainer_usesConfigGradClipNorm_toLimitUpdateMagnitude() throws IOException {
        Path tmp = Files.createTempFile("deepj_lm_clip", ".txt");
        Files.writeString(tmp, "hello hello hello hello hello hello");
        double deltaUnclipped = parameterDelta(tmp, 1.0f);
        double deltaClipped = parameterDelta(tmp, 1e-6f);
        Assertions.assertTrue(deltaUnclipped > 0.0f);
        Assertions.assertTrue(deltaClipped > 0.0f);
        Assertions.assertTrue(deltaClipped < deltaUnclipped,
                "Expected clipped run to update less than unclipped run");
    }

    private static double parameterDelta(Path corpus, float clipNorm) throws IOException {
        TextDataset dataset = TextDataset.fromFile(corpus, new ByteTokenizer(), 8, 7L);
        GPTConfig config = new GPTConfig(
                ByteTokenizer.VOCAB_SIZE, 8, 32, 4, 1, 64, 1.0f, clipNorm);
        GPTModel model = new GPTModel(config, 2L);
        Parameter parameter = model.parameters().get(0);
        Tensor before = parameter.value.multiplyScalar(1.0f);
        CausalLMTraining.trainer(model, dataset, 1e-2f).trainStep(2);
        return parameter.value.subtract(before).sumAbs();
    }

    private static float accumulatedLoss(GPTModel model) throws IOException {
        var optimizer = AdamW.defaultAdamW(1e-2f);
        return CausalLMTraining.trainer(model, accumulationDataset(), optimizer, 2).trainStep(1);
    }

    private static float batchedLoss(GPTModel model) throws IOException {
        return CausalLMTraining.trainer(model, accumulationDataset(), 1e-2f).trainStep(2);
    }

    private static TextDataset accumulationDataset() throws IOException {
        return tinyDataset("one two three four five six seven eight nine ten", 8);
    }

    private static GPTModel gptModel(long seed) {
        return new GPTModel(new GPTConfig(ByteTokenizer.VOCAB_SIZE, 8, 32, 4, 1, 64), seed);
    }

    private static void assertParametersClose(List<Parameter> expected, List<Parameter> actual) {
        Assertions.assertEquals(expected.size(), actual.size());
        for (int index = 0; index < expected.size(); index++) {
            assertTensorClose(expected.get(index).value, actual.get(index).value);
        }
    }

    private static void assertTensorClose(Tensor expected, Tensor actual) {
        Assertions.assertArrayEquals(expected.data, actual.data, 1e-6f);
    }
}
