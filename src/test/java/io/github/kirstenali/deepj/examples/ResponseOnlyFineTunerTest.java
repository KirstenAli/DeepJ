package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.data.ResponseOnlyTextDataset;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ResponseOnlyFineTunerTest {

    @TempDir
    Path temporaryDirectory;

    @BeforeEach
    void useCpuBackend() {
        Tensor.setBackend(new CpuBackend());
    }

    @Test
    void fineTunesOneReusableSource() throws Exception {
        Path base = trainBaseModel();
        Path corpus = writeResponseCorpus();
        Path output = temporaryDirectory.resolve("fine-tuned");

        var result = ResponseOnlyFineTuner.run(config(base, output, corpus));

        assertEquals(1, result.steps());
        assertTrue(Files.isRegularFile(output.resolve("model-final.dj")));
        assertTrue(Files.isRegularFile(output.resolve("training-latest.dj")));
        assertTrue(Files.isRegularFile(output.resolve("fine-tuning.properties")));
    }

    private Path trainBaseModel() throws Exception {
        Path corpus = Files.writeString(temporaryDirectory.resolve("base.txt"), baseText());
        Path output = temporaryDirectory.resolve("base");
        TrainDeepSeekTinyStories.run(baseConfig(corpus, output));
        return output;
    }

    private DeepSeekTinyStoriesConfig baseConfig(Path corpus, Path output) {
        var files = new DeepSeekTinyStoriesConfig.FilesConfig(corpus, output, null);
        var architecture = new DeepSeekTinyStoriesConfig.Architecture(
                64, 8, 2, 1, 16, 4, 2, 0.2f, 1.0f);
        var training = new DeepSeekTinyStoriesConfig.Training(
                1, 1, 1e-2f, 1e-3f, 0, 1, 1, 0);
        return new DeepSeekTinyStoriesConfig(files, architecture,
                training, new DeepSeekTinyStoriesConfig.TokenizerConfig(280, 1), 42L);
    }

    private ResponseFineTuningConfig config(Path base, Path output, Path corpus) {
        var files = new ResponseFineTuningConfig.FilesConfig(
                base, output, base.resolve("model-final.dj"), null);
        var training = new ResponseFineTuningConfig.Training(
                1, 1, 1e-3f, 1e-4f, 0, 1, 1, 0);
        var source = new ResponseOnlyTextDataset.Source(corpus, 1);
        return new ResponseFineTuningConfig(files, training, List.of(source), 43L);
    }

    private Path writeResponseCorpus() throws Exception {
        String value = "Instruction:\nHello\nResponse:\nHello there.\n<|endoftext|>\n";
        return Files.writeString(temporaryDirectory.resolve("responses.txt"), value);
    }

    private static String baseText() {
        return "Hello there. How can I help you today? <|endoftext|>\n".repeat(100);
    }
}
