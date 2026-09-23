package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.training.TrainingResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TrainDeepJPrismTinyStoriesTest {

    @TempDir
    Path temporaryDirectory;

    @BeforeEach
    void useCpuBackend() {
        Tensor.setBackend(new CpuBackend());
    }

    @Test
    void endToEndRunWritesReusableArtifacts() throws Exception {
        Path corpus = temporaryDirectory.resolve("stories.txt");
        Files.writeString(corpus, tinyStories());
        Path output = temporaryDirectory.resolve("output");
        DeepJPrismTinyStoriesConfig config = config(corpus, output);

        TrainingResult result = TrainDeepJPrismTinyStories.run(config);

        assertEquals(3, result.steps());
        assertTrue(Float.isFinite(result.lastLoss()));
        assertTrue(Files.isRegularFile(output.resolve("tokenizer.bpe")));
        assertTrue(Files.isRegularFile(output.resolve("model-latest.dj")));
        assertTrue(Files.isRegularFile(output.resolve("training-latest.dj")));
        assertTrue(Files.isRegularFile(output.resolve("model-final.dj")));
        assertTrue(Files.isRegularFile(output.resolve("training.properties")));
        assertResumeIsExact(corpus, output);
        assertExportIsReloadable(corpus, output);
    }

    private void assertResumeIsExact(Path corpus, Path output) throws Exception {
        Path finalModel = output.resolve("model-final.dj");
        byte[] expected = Files.readAllBytes(finalModel);
        TrainingResult result = TrainDeepJPrismTinyStories.run(resumeConfig(corpus, output));
        assertEquals(3, result.steps());
        assertArrayEquals(expected, Files.readAllBytes(finalModel));
    }

    private DeepJPrismTinyStoriesConfig resumeConfig(Path corpus, Path output) {
        DeepJPrismTinyStoriesConfig base = config(corpus, output);
        var files = new DeepJPrismTinyStoriesConfig.FilesConfig(
                corpus, output, output.resolve("training-latest.dj"));
        return new DeepJPrismTinyStoriesConfig(files, base.architecture(), base.training(),
                base.tokenizer(), base.seed());
    }

    private void assertExportIsReloadable(Path corpus, Path output) throws Exception {
        Path bundle = temporaryDirectory.resolve("bundle");
        Path checkpoint = output.resolve("model-final.dj");
        assertDoesNotThrow(() -> ExportDeepJPrismTinyStories.export(
                output, checkpoint, bundle, corpus, 1));
        assertTrue(Files.isRegularFile(bundle.resolve("model.dj")));
        assertTrue(Files.isRegularFile(bundle.resolve("tokenizer.bpe")));
        assertTrue(Files.isRegularFile(bundle.resolve("config.json")));
        assertTrue(Files.isRegularFile(bundle.resolve("README.md")));
        String modelCard = Files.readString(bundle.resolve("README.md"));
        assertTrue(modelCard.contains("# DeepJ TinyStories"));
        assertFalse(modelCard.contains("causal language model"));
    }

    @Test
    void configRejectsInvalidScheduleAndTokenizer() {
        assertThrows(IllegalArgumentException.class,
                () -> new DeepJPrismTinyStoriesConfig.Training(2, 1, 1e-3f,
                        1e-4f, 2, 1, 1, 0));
        assertThrows(IllegalArgumentException.class,
                () -> new DeepJPrismTinyStoriesConfig.TokenizerConfig(261, 1));
    }

    private DeepJPrismTinyStoriesConfig config(Path corpus, Path output) {
        var files = new DeepJPrismTinyStoriesConfig.FilesConfig(corpus, output, null);
        var architecture = new DeepJPrismTinyStoriesConfig.Architecture(
                8, 8, 2, 1, 16, 4, 2, 0.2f, 1.0f);
        var training = new DeepJPrismTinyStoriesConfig.Training(
                3, 1, 1e-2f, 1e-3f, 1, 1, 2, 0);
        var tokenizer = new DeepJPrismTinyStoriesConfig.TokenizerConfig(280, 1);
        return new DeepJPrismTinyStoriesConfig(files, architecture, training, tokenizer, 42L);
    }

    private static String tinyStories() {
        String story = "Once there was a little cat who liked the bright red garden.\n"
                + "The cat met a dog and they played together all day.\n"
                + "<|endoftext|>\n";
        return story.repeat(100);
    }
}
