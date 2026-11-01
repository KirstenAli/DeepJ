package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.MetalUnavailableClassLoader;
import io.github.kirstenali.deepj.nn.transformer.nova.DeepJNova;
import io.github.kirstenali.deepj.nn.transformer.nova.DeepJNovaConfig;
import io.github.kirstenali.deepj.nn.transformer.prism.DeepJPrism;
import io.github.kirstenali.deepj.nn.transformer.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETrainer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.api.function.Executable;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.lang.reflect.InvocationTargetException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EvaluationExamplesTest {

    private static final String RECORD = "Instruction:\nWhat is one plus one?\nResponse:\n2.\n<|endoftext|>\n\n";
    @TempDir Path directory;
    private Properties previousProperties;
    private TensorBackend previousBackend;
    private int vocabulary;

    @BeforeEach
    void prepareIsolatedArtifacts() throws Exception {
        previousProperties = (Properties) System.getProperties().clone();
        previousBackend = Tensor.backend();
        Tensor.setBackend(new CpuBackend());
        var tokenizer = new BPETrainer().train(RECORD.repeat(20), 280, List.of("<|endoftext|>"));
        vocabulary = tokenizer.vocabSize();
        BPEModelIO.save(directory.resolve("tokenizer.bpe"), tokenizer);
        Files.writeString(directory.resolve("validation.txt"), RECORD.repeat(20));
        Files.writeString(directory.resolve("corpus.txt"), RECORD.repeat(30_000));
        configureEvaluation();
    }

    private void configureEvaluation() {
        Map.of("deepj.base", directory.toString(), "deepj.checkpoint", directory.resolve("model.dj").toString(),
                "deepj.corpus", directory.resolve("corpus.txt").toString(),
                "deepj.validation", directory.resolve("validation.txt").toString(),
                "deepj.alpacaValidation", directory.resolve("validation.txt").toString(),
                "deepj.factValidation", directory.resolve("validation.txt").toString(),
                "deepj.evalBatches", "1", "deepj.generateTokens", "1", "deepj.validationMiB", "1",
                "deepj.exactSamples", "1").forEach(System::setProperty);
    }

    @AfterEach
    void restoreProcessState() {
        Tensor.backend().releaseResources();
        Tensor.setBackend(previousBackend);
        System.setProperties(previousProperties);
    }

    @Test
    void bothPretrainingEvaluatorsReadEmbeddedModelSettingsAndReportFiniteLoss() throws Throwable {
        var model = new DeepJNova(new DeepJNovaConfig(vocabulary, 64, 8, 4, 2, 1, 12), 42L);
        model.save(directory.resolve("model.dj"));
        assertFiniteValidation(capture(() -> EvaluateDeepJ100M.main(new String[0])));
        assertFiniteValidation(capture(() -> EvaluateDeepJ90MPretraining.main(new String[0])));
    }

    @Test
    void instructionEvaluatorChecksResponseOnlyLossAndGeneratesSamples() throws Throwable {
        savePrism();
        String output = capture(() -> EvaluateDeepJ90M.main(new String[0]));
        assertFiniteValidation(output);
        assertTrue(output.contains("=== Sample ==="));
        assertTrue(output.contains("Instruction:"));
    }

    @Test
    void knowledgeEvaluatorSupportsOptionalSamples() throws Throwable {
        savePrism();
        for (boolean samples : new boolean[]{false, true}) {
            System.setProperty("deepj.generateSamples", Boolean.toString(samples));
            String output = capture(() -> EvaluateDeepJPrismKnowledge.main(new String[0]));
            assertTrue(output.contains("Alpaca validation:"));
            assertTrue(output.contains("Facts validation:"));
            assertTrue(output.contains("Exact arithmetic accuracy:"));
            assertTrue(output.contains("=== Sample ===") == samples);
        }
    }

    @Test
    void syntheticBenchmarkExercisesSmallTrainingStepsWithoutPrivateData() throws Throwable {
        configureBenchmark();
        String output = capture(() -> BenchmarkDeepJ100M.main(new String[0]));
        assertTrue(output.contains("parameters=636"));
        assertTrue(output.contains("measured round=0"));
        assertFalse(output.contains("loss=NaN"));
        assertFalse(output.contains("loss=Infinity"));
    }

    @Test
    void benchmarkRejectsAnInvalidStepBudget() throws Throwable {
        configureBenchmark();
        System.setProperty("deepj.benchmarkSteps", "0");
        capture(() -> assertThrows(IllegalArgumentException.class,
                () -> BenchmarkDeepJ100M.main(new String[0])));
    }

    @Test
    void benchmarkDoesNotSilentlyUseCpuUnlessExplicitlyAllowed() throws Exception {
        System.clearProperty("deepj.allowCpuBenchmark");
        try (var isolated = new MetalUnavailableClassLoader()) {
            var main = isolated.method("examples.BenchmarkDeepJ100M", "main", String[].class);
            var failure = assertThrows(InvocationTargetException.class,
                    () -> main.invoke(null, (Object) new String[0]));
            assertInstanceOf(IllegalStateException.class, failure.getCause());
            assertTrue(failure.getCause().getMessage().contains("Metal is unavailable"));
        }
    }

    private void configureBenchmark() {
        Map.of("deepj.allowCpuBenchmark", "true", "deepj.vocabSize", "16", "deepj.seqLen", "4",
                "deepj.dModel", "8", "deepj.heads", "4", "deepj.kvHeads", "2", "deepj.layers", "1",
                "deepj.dFF", "12", "deepj.benchmarkSteps", "1").forEach(System::setProperty);
    }

    private void savePrism() throws Exception {
        var config = new DeepJPrismConfig(vocabulary, 64, 8, 2, 1, 12, 4, 4);
        new DeepJPrism(config, 42L).save(directory.resolve("model.dj"));
    }

    private static String capture(Executable action) throws Throwable {
        var bytes = new ByteArrayOutputStream();
        PrintStream previous = System.out;
        try (var output = new PrintStream(bytes, true, StandardCharsets.UTF_8)) {
            System.setOut(output);
            action.execute();
        } finally {
            System.setOut(previous);
        }
        return bytes.toString(StandardCharsets.UTF_8);
    }

    private static void assertFiniteValidation(String output) {
        assertTrue(output.contains("Validation: tokens="));
        assertFalse(output.contains("loss=NaN"));
        assertFalse(output.contains("perplexity=Infinity"));
    }
}
