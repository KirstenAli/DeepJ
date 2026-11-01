package io.github.kirstenali.deepj.cli;

import io.github.kirstenali.deepj.nn.transformer.DeepJChat;
import io.github.kirstenali.deepj.persistence.ModelSerializer;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.testing.TypedChatFixtures;
import java.io.*;
import java.nio.file.*;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import static org.junit.jupiter.api.Assertions.*;

class TypedCliGenerationTest {
    @TempDir Path directory;
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest
    @MethodSource("io.github.kirstenali.deepj.testing.NdModelFixtures#typedCases")
    void actualCommandLoadsTypedFilesAndKeepsInstructionFormattingAndCpuSelection(String family, DType dtype)
            throws IOException {
        Path file = TypedChatFixtures.save(family, dtype, directory);
        String expected = DeepJChat.load(file).generate("Hello", 4, 0.8f, 40, 1234, 1.1f);
        var output = new StringWriter();
        var errors = new StringWriter();
        var services = services(output, errors);
        assertEquals(0, DeepJCli.execute(services, "generate", file.toString(), "--backend", "cpu",
                "--prompt", "Hello", "--max-tokens", "4", "--temperature", "0.8", "--top-k", "40",
                "--seed", "1234", "--repetition-penalty", "1.1"));
        String safe = CliText.safe(expected);
        assertEquals(safe.endsWith("\n") ? safe : safe + System.lineSeparator(), output.toString());
        assertEquals("", errors.toString());
        assertInstanceOf(CpuBackend.class, Tensor.backend());
    }

    private static CliServices services(StringWriter output, StringWriter errors) {
        return new CliServices(InputStream.nullInputStream(), new PrintWriter(output, true),
                new PrintWriter(errors, true), false, Map.of(), CliModel::load, CliConsole::system,
                path -> ModelSerializer.inspect(CliModel.modelFile(path)), () -> false);
    }
}
