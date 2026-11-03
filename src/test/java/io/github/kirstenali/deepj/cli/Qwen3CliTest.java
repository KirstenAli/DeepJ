package io.github.kirstenali.deepj.cli;

import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3ChatFixture;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.io.ByteArrayInputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

class Qwen3CliTest {
    @TempDir Path directory;

    @Test
    @org.junit.jupiter.api.condition.EnabledIfSystemProperty(named = "deepj.qwen3.pretrained", matches = ".+")
    void normalCliCommandLoadsTheFullOfficialModelOnMetal() {
        StringWriter output = new StringWriter(), errors = new StringWriter();
        try {
            int result = DeepJCli.execute(services(output, errors), "--backend", "metal", "generate",
                    System.getProperty("deepj.qwen3.pretrained"), "--prompt", "Reply with one word: hello",
                    "--max-tokens", "8", "--top-k", "1");
            assertEquals(0, result, errors.toString());
            assertEquals("Hello." + System.lineSeparator(), output.toString());
        } finally {
            Tensor.setBackend(new CpuBackend());
        }
    }

    @ParameterizedTest @ValueSource(strings = {".", "config.json", "model.safetensors"})
    void normalGenerateCommandLoadsQwenAndPrintsOnlyTheAnswer(String file) throws Exception {
        Tensor.setBackend(new CpuBackend());
        Path snapshot = Qwen3ChatFixture.create(directory.resolve("model"));
        StringWriter output = new StringWriter(), errors = new StringWriter();
        int result = DeepJCli.execute(services(output, errors), "--backend", "cpu", "generate",
                snapshot.resolve(file).toString(), "--prompt", "hello", "--max-tokens", "4", "--top-k", "1");
        assertEquals(0, result, errors.toString());
        assertEquals("xxxx" + System.lineSeparator(), output.toString());
        assertEquals("", errors.toString());
    }

    @Test void inspectReadsQwenMetadataWithoutChangingTheBackend() throws Exception {
        Tensor.setBackend(new CpuBackend());
        Path snapshot = Qwen3ChatFixture.create(directory.resolve("model"));
        var backend = Tensor.backend();
        StringWriter output = new StringWriter(), errors = new StringWriter();
        assertEquals(0, DeepJCli.execute(services(output, errors), "inspect", snapshot.toString()));
        assertSame(backend, Tensor.backend());
        assertTrue(output.toString().contains("Architecture    Qwen3"));
        assertTrue(output.toString().contains("Prompt format   Qwen3 (non-thinking)"));
    }

    static CliServices services(StringWriter output, StringWriter errors) {
        var defaults = CliServices.system();
        return new CliServices(new ByteArrayInputStream(new byte[0]), new PrintWriter(output, true),
                new PrintWriter(errors, true), false, Map.of(), defaults.models(), defaults.consoles(),
                defaults.inspector(), defaults.metalAvailable());
    }
}
