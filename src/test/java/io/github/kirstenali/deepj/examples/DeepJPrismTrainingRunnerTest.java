package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.models.ModelLoader;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.training.TrainingResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DeepJPrismTrainingRunnerTest {

    @TempDir
    Path temporaryDirectory;

    @BeforeEach
    void useCpuBackend() {
        Tensor.setBackend(new CpuBackend());
    }

    @Test
    void endToEndRunWritesReusableArtifacts() throws Exception {
        Path corpus = temporaryDirectory.resolve("corpus.txt");
        Files.writeString(corpus, corpusText());
        Path output = temporaryDirectory.resolve("output");
        DeepJPrismTrainingConfig config = config(corpus, output);

        TrainingResult result = DeepJPrismTrainingRunner.runSequential(config);

        assertEquals(3, result.steps());
        assertTrue(Float.isFinite(result.lastLoss()));
        assertTrue(Files.isRegularFile(output.resolve("tokenizer.bpe")));
        assertTrue(Files.isRegularFile(output.resolve("model-latest.dj")));
        assertTrue(Files.isRegularFile(output.resolve("training-latest.dj")));
        assertTrue(Files.isRegularFile(output.resolve("model-final.dj")));
        assertEquals(corpus.toAbsolutePath().toString(),
                ModelLoader.load(output.resolve("model-final.dj")).metadata().get("corpus"));
        assertFalse(Files.exists(output.resolve("training.properties")));
        assertResumeIsExact(corpus, output);
    }

    private void assertResumeIsExact(Path corpus, Path output) throws Exception {
        Path finalModel = output.resolve("model-final.dj");
        byte[] expected = Files.readAllBytes(finalModel);
        TrainingResult result = DeepJPrismTrainingRunner.runSequential(resumeConfig(corpus, output));
        assertEquals(3, result.steps());
        assertArrayEquals(expected, Files.readAllBytes(finalModel));
    }

    private DeepJPrismTrainingConfig resumeConfig(Path corpus, Path output) {
        DeepJPrismTrainingConfig base = config(corpus, output);
        var files = new DeepJPrismTrainingConfig.FilesConfig(
                corpus, output, output.resolve("training-latest.dj"));
        return new DeepJPrismTrainingConfig(files, base.architecture(), base.training(),
                base.tokenizer(), base.seed());
    }

    @Test
    void configRejectsInvalidScheduleAndTokenizer() {
        assertThrows(IllegalArgumentException.class,
                () -> new DeepJPrismTrainingConfig.Training(2, 1, 1e-3f,
                        1e-4f, 2, 1, 1, 0));
        assertThrows(IllegalArgumentException.class,
                () -> new DeepJPrismTrainingConfig.TokenizerConfig(261, 1));
    }

    private DeepJPrismTrainingConfig config(Path corpus, Path output) {
        var files = new DeepJPrismTrainingConfig.FilesConfig(corpus, output, null);
        var architecture = new DeepJPrismTrainingConfig.Architecture(
                8, 8, 2, 1, 16, 4, 2, 0.2f, 1.0f);
        var training = new DeepJPrismTrainingConfig.Training(
                3, 1, 1e-2f, 1e-3f, 1, 1, 2, 0);
        var tokenizer = new DeepJPrismTrainingConfig.TokenizerConfig(280, 1);
        return new DeepJPrismTrainingConfig(files, architecture, training, tokenizer, 42L);
    }

    private static String corpusText() {
        String story = "Once there was a little cat who liked the bright red garden.\n"
                + "The cat met a dog and they played together all day.\n"
                + "<|endoftext|>\n";
        return story.repeat(100);
    }
}
