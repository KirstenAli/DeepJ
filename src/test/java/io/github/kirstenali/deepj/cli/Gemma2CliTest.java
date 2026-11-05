package io.github.kirstenali.deepj.cli;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import io.github.kirstenali.deepj.tensor.DType;
import java.io.StringWriter;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

class Gemma2CliTest {
    private static final Path ROOT = Path.of("src/test/resources/gemma2");

    @ParameterizedTest @ValueSource(strings = {".", "config.json", "model.safetensors"})
    void existingGenerateCommandRoutesGemmaAndPrintsOnlyItsReply(String selection) throws Exception {
        var output = new StringWriter();
        var errors = new StringWriter();
        int exit = DeepJCli.execute(Qwen3CliTest.services(output, errors), "--backend", "cpu", "generate",
                ROOT.resolve(selection).toString(), "--prompt", "Hello", "--max-tokens", "4", "--top-k", "1");
        assertEquals(0, exit, errors.toString());
        assertEquals(CliText.safe(reply()) + System.lineSeparator(), output.toString());
    }

    @Test void gemmaChatLoadsExplicitFp32AndUsesTheSharedStreamingAdapter() throws Exception {
        try (var loaded = CliModel.load(ROOT, BackendChoice.CPU, DType.FLOAT32)) {
            assertEquals("Gemma 2", loaded.name());
            assertEquals("CPU", loaded.backend());
            var partial = new java.util.ArrayList<String>();
            assertEquals(reply(), loaded.stream("Hello", new SamplingSettings(4, 1, 1, 2, 1), partial::add));
            assertFalse(partial.isEmpty());
        }
    }

    @Test void inspectRecognizesGemmaWithoutAllocatingWeightsOrChangingBackend() throws Exception {
        var backend = io.github.kirstenali.deepj.tensor.Tensor.backend();
        var details = CliModel.inspect(ROOT);
        assertSame(backend, io.github.kirstenali.deepj.tensor.Tensor.backend());
        assertEquals(24, details.tensors());
        assertEquals("Gemma 2", details.settings().metadata().get("name"));
    }

    private static String reply() throws Exception {
        return Qwen3Json.string(Qwen3Json.read(ROOT.resolve("tokenizer-reference.json")).get("chat_reply"));
    }
}
