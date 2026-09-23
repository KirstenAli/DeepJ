package io.github.kirstenali.deepj.training;

import io.github.kirstenali.deepj.data.TextDataset;
import io.github.kirstenali.deepj.models.CausalLM;
import io.github.kirstenali.deepj.models.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.models.prism.DeepJPrism;
import io.github.kirstenali.deepj.models.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.models.origin.DeepJOrigin;
import io.github.kirstenali.deepj.models.orbit.DeepJOrbitConfig;
import io.github.kirstenali.deepj.models.orbit.DeepJOrbit;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.optimisers.ParameterOptimizer;
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
import java.util.ArrayList;
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
                new DeepJOrigin(new DeepJOriginConfig(vocab, 8, 32, 4, 1, 64), 1L),
                new DeepJOrbit(new DeepJOrbitConfig(vocab, 8, 32, 4, 1, DeepJOrbitConfig.defaultDFF(32)), 1L),
                new DeepJPrism(new DeepJPrismConfig(vocab, 8, 32, 4, 1, 64, 16, 8), 1L)
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

        DeepJOriginConfig cfg = new DeepJOriginConfig(ByteTokenizer.VOCAB_SIZE, 8, 32, 4, 1, 64);
        DeepJOrigin model = new DeepJOrigin(cfg, 2L);

        Trainer trainer = CausalLMTraining.trainer(model, ds, 1e-2f);

        double loss = trainer.trainStep(2);
        Assertions.assertTrue(Double.isFinite(loss));
        Assertions.assertTrue(loss > 0.0f);
    }

    @Test
    void batchingAveragesIndividualLossesAndGradients() throws IOException {
        TrainingObservation batched = observeTraining(2, 1);
        TrainingObservation singles = observeTraining(1, 2);

        Assertions.assertEquals(mean(singles.losses()), batched.losses().get(0), 1e-6f);
        assertGradientAverage(batched.gradients().get(0), singles.gradients());
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
        DeepJOriginConfig config = new DeepJOriginConfig(
                ByteTokenizer.VOCAB_SIZE, 8, 32, 4, 1, 64, 1.0f, clipNorm);
        DeepJOrigin model = new DeepJOrigin(config, 2L);
        Parameter parameter = model.parameters().get(0);
        Tensor before = parameter.value.multiplyScalar(1.0f);
        CausalLMTraining.trainer(model, dataset, 1e-2f).trainStep(2);
        return parameter.value.subtract(before).sumAbs();
    }

    private static TrainingObservation observeTraining(int batchSize, int steps) throws IOException {
        GradientRecorder optimizer = new GradientRecorder();
        Trainer trainer = CausalLMTraining.trainer(averagingModel(), averagingDataset(), optimizer);
        List<Float> losses = new ArrayList<>();
        for (int step = 0; step < steps; step++) {
            losses.add(trainer.trainStep(batchSize));
        }
        return new TrainingObservation(losses, optimizer.gradients);
    }

    private static DeepJOrigin averagingModel() {
        DeepJOriginConfig config = new DeepJOriginConfig(
                ByteTokenizer.VOCAB_SIZE, 8, 32, 4, 1, 64, 0.2f, Float.MAX_VALUE);
        return new DeepJOrigin(config, 3L);
    }

    private static TextDataset averagingDataset() throws IOException {
        return tinyDataset("one two three four five six seven eight nine ten", 8);
    }

    private static float mean(List<Float> values) {
        return (values.get(0) + values.get(1)) * 0.5f;
    }

    private static void assertGradientAverage(List<Tensor> batched,
                                              List<List<Tensor>> singles) {
        for (int index = 0; index < batched.size(); index++) {
            assertTensorAverage(batched.get(index), singles.get(0).get(index),
                    singles.get(1).get(index));
        }
    }

    private static void assertTensorAverage(Tensor actual, Tensor first, Tensor second) {
        for (int index = 0; index < actual.data.length; index++) {
            float expected = (first.data[index] + second.data[index]) * 0.5f;
            Assertions.assertEquals(expected, actual.data[index], 1e-6f);
        }
    }

    private record TrainingObservation(List<Float> losses,
                                       List<List<Tensor>> gradients) {}

    private static final class GradientRecorder implements ParameterOptimizer {

        private final List<List<Tensor>> gradients = new ArrayList<>();

        @Override
        public void step(List<Parameter> parameters) {
            gradients.add(parameters.stream().map(parameter -> new Tensor(parameter.grad)).toList());
        }
    }
}
