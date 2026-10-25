package io.github.kirstenali.deepj.cli;

import io.github.kirstenali.deepj.chatui.DeepJChatService;
import io.github.kirstenali.deepj.models.PromptFormat;
import io.github.kirstenali.deepj.models.TextGenerator;
import io.github.kirstenali.deepj.models.prism.DeepJPrism;
import io.github.kirstenali.deepj.models.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETokenizer;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETrainer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CliPromptFormatTest {
    private static final SamplingSettings SETTINGS = new SamplingSettings(6, 0.8f, 20, 1234, 1.1f);
    @TempDir Path directory;
    private TensorBackend previous;
    private BPETokenizer tokenizer;

    @BeforeEach void selectCpu() {
        previous = Tensor.backend();
        Tensor.setBackend(new CpuBackend());
    }

    @AfterEach void restore() {
        Tensor.backend().releaseResources();
        Tensor.setBackend(previous);
    }

    @ParameterizedTest
    @ValueSource(strings = {"Hello", "hey", "What is a computer?", "\n  Hello  \n"})
    void instructionModelsMatchTheDesktopAndExactFriskyTemplate(String prompt) throws Exception {
        verify(prompt, "Instruction:\n" + prompt.strip() + "\nResponse:", PromptFormat.INSTRUCTION_RESPONSE);
    }

    @Test
    void alreadyFormattedPromptsArePreservedWithoutADuplicateWrapper() throws Exception {
        String formatted = "Instruction:\nHello\nResponse:";
        verify(formatted, formatted, PromptFormat.INSTRUCTION_RESPONSE);
    }

    @Test
    void plainModelsDoNotReceiveAnInstructionWrapper() throws Exception {
        verify("Hello", "Hello", PromptFormat.PLAIN);
    }

    private void verify(String prompt, String formatted, PromptFormat format) throws Exception {
        DeepJPrism model = checkpoint(format);
        String prefix = tokenizer.decode(tokenizer.encode(formatted));
        String full = TextGenerator.generate(model, tokenizer, model.config(), formatted,
                SETTINGS.maxTokens(), SETTINGS.temperature(), SETTINGS.topK(), SETTINGS.seed(),
                SETTINGS.repetitionPenalty());
        assertEquals(formatted, prefix);
        assertResponses(prompt, full.substring(prefix.length()), format);
    }

    private void assertResponses(String prompt, String expected, PromptFormat format) throws Exception {
        var desktop = new DeepJChatService();
        desktop.loadModel(directory.resolve("model.dj"));
        var desktopUpdates = new ArrayList<String>();
        String desktopText = desktop.generate(prompt, SETTINGS.maxTokens(), SETTINGS.temperature(),
                SETTINGS.topK(), SETTINGS.seed(), SETTINGS.repetitionPenalty(), desktopUpdates::add);
        try (var cli = CliModel.load(directory, BackendChoice.CPU)) {
            var cliUpdates = new ArrayList<String>();
            assertEquals(format, cli.inspection().settings().promptFormat());
            assertEquals(expected, cli.stream(prompt, SETTINGS, cliUpdates::add));
            assertEquals(expected, desktopText);
            assertEquals(desktopUpdates, cliUpdates);
        }
    }

    private DeepJPrism checkpoint(PromptFormat format) throws IOException {
        var bpe = new BPETrainer().trainTokenizerWithDefaults("Hello hello world deepj computer", 280).model();
        tokenizer = new BPETokenizer(bpe);
        BPEModelIO.save(directory.resolve("tokenizer.bpe"), bpe);
        var config = new DeepJPrismConfig(bpe.vocabSize(), 128, 8, 2, 1, 16, 4, 2);
        var model = new DeepJPrism(config, 42);
        model.setPromptFormat(format);
        model.save(directory.resolve("model.dj"));
        return model;
    }
}
