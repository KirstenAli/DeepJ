package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.training.TrainingResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TrainDeepSeekTinyStoriesTest {

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
        DeepSeekTinyStoriesConfig config = config(corpus, output);

        TrainingResult result = TrainDeepSeekTinyStories.run(config);

        assertEquals(3, result.steps());
        assertTrue(Float.isFinite(result.lastLoss()));
        assertTrue(Files.isRegularFile(output.resolve("tokenizer.bpe")));
        assertTrue(Files.isRegularFile(output.resolve("model-latest.dj")));
        assertTrue(Files.isRegularFile(output.resolve("model-final.dj")));
        assertTrue(Files.isRegularFile(output.resolve("training.properties")));
        assertExportIsReloadable(corpus, output);
    }

    private void assertExportIsReloadable(Path corpus, Path output) throws Exception {
        Path bundle = temporaryDirectory.resolve("bundle");
        Path checkpoint = output.resolve("model-final.dj");
        assertDoesNotThrow(() -> ExportDeepSeekTinyStories.export(
                output, checkpoint, bundle, corpus, 1));
        assertTrue(Files.isRegularFile(bundle.resolve("model.dj")));
        assertTrue(Files.isRegularFile(bundle.resolve("tokenizer.bpe")));
        assertTrue(Files.isRegularFile(bundle.resolve("config.json")));
        assertTrue(Files.isRegularFile(bundle.resolve("README.md")));
    }

    @Test
    void configRejectsInvalidScheduleAndTokenizer() {
        assertThrows(IllegalArgumentException.class,
                () -> new DeepSeekTinyStoriesConfig.Training(2, 1, 1e-3f,
                        1e-4f, 2, 1, 1, 0));
        assertThrows(IllegalArgumentException.class,
                () -> new DeepSeekTinyStoriesConfig.TokenizerConfig(261, 1));
    }

    private DeepSeekTinyStoriesConfig config(Path corpus, Path output) {
        var files = new DeepSeekTinyStoriesConfig.FilesConfig(corpus, output, null);
        var architecture = new DeepSeekTinyStoriesConfig.Architecture(
                8, 8, 2, 1, 16, 4, 2, 0.2f, 1.0f);
        var training = new DeepSeekTinyStoriesConfig.Training(
                3, 1, 1e-2f, 1e-3f, 1, 1, 2, 0);
        var tokenizer = new DeepSeekTinyStoriesConfig.TokenizerConfig(280, 1);
        return new DeepSeekTinyStoriesConfig(files, architecture, training, tokenizer, 42L);
    }

    private static String tinyStories() {
        String story = "Once there was a little cat who liked the bright red garden.\n"
                + "The cat met a dog and they played together all day.\n"
                + "<|endoftext|>\n";
        return story.repeat(100);
    }
}
