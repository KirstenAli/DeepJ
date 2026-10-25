package io.github.kirstenali.deepj.cli;

import io.github.kirstenali.deepj.models.ContextWindowMode;

import io.github.kirstenali.deepj.models.DeepJChat;
import io.github.kirstenali.deepj.models.prism.DeepJPrism;
import io.github.kirstenali.deepj.models.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.*;

class CliModelTest {
    @TempDir Path directory;
    private TensorBackend previous;

    @Test
    void terminalGenerationUsesTheSelectedRollingMode() throws IOException {
        var mode = ContextWindowMode.ROLLING;
        Path path = save(256);
        var settings = new SamplingSettings(12, 0.8f, 20, 47, 1.1f, mode);
        try (var model = CliModel.load(path, BackendChoice.CPU)) {
            String prompt = "A prompt long enough to fill this small context window.";
            assertEquals(DeepJChat.load(directory).withContextWindow(mode)
                    .generate(prompt, 12, 0.8f, 20, 47, 1.1f), model.stream(prompt, settings, text -> {}));
        }
    }

    @BeforeEach void selectCpu() {
        previous = Tensor.backend();
        Tensor.setBackend(new CpuBackend());
    }

    @AfterEach void restore() {
        Tensor.backend().releaseResources();
        Tensor.setBackend(previous);
    }

    @Test
    void directoryAndExplicitCheckpointPathsLoadTheSameEmbeddedConfiguration() throws IOException {
        Path path = save(256);
        var updates = new ArrayList<String>();
        var settings = new SamplingSettings(3, 0.8f, 40, 1234, 1);
        try (var model = CliModel.load(path, BackendChoice.CPU)) {
            String text = model.stream("Hello", settings, updates::add);
            assertEquals(DeepJChat.load(directory).generate("Hello", 3, 0.8f, 40, 1234), text);
            assertEquals(text, updates.get(updates.size() - 1));
            assertEquals("CPU", model.backend());
            assertEquals("DeepJ Prism", model.name());
            assertTrue(model.inspection().parameters() > 0);
        }
        assertEquals(path, CliModel.modelFile(directory));
        assertEquals(path, CliModel.modelFile(path));
    }

    @Test
    void missingTokenizerReportsAnErrorAndReleasesTheSelectedBackend() throws IOException {
        Path path = save(280);
        IOException failure = assertThrows(IOException.class, () -> CliModel.load(path, BackendChoice.CPU));
        assertTrue(failure.getMessage().contains("tokenizer.bpe"));
    }

    private Path save(int vocabulary) throws IOException {
        var model = new DeepJPrism(new DeepJPrismConfig(vocabulary, 32, 8, 2, 1, 16, 4, 2), 42);
        Path path = directory.resolve("model.dj");
        model.save(path);
        return path;
    }
}
