package io.github.kirstenali.deepj.cli;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.io.StringWriter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import static org.junit.jupiter.api.Assertions.*;

@EnabledIfSystemProperty(named = "deepj.gemma2Checkpoint", matches = ".+")
class Gemma2PretrainedCliTest {
    @Test void realGemmaRespondsToAPromptThroughTheExistingCliCommand() {
        var answer = new StringWriter();
        var errors = new StringWriter();
        try {
            int exit = DeepJCli.execute(Qwen3CliTest.services(answer, errors), "--backend", "metal", "generate",
                    System.getProperty("deepj.gemma2Checkpoint"), "--prompt", "Reply with one word: hello",
                    "--max-tokens", "12", "--top-k", "1");
            assertEquals(0, exit, errors.toString());
            assertTrue(answer.toString().toLowerCase(java.util.Locale.ROOT).contains("hello"), answer.toString());
            System.out.println("Gemma CLI reply: " + answer.toString().strip());
        } finally {
            Tensor.backend().releaseResources();
            Tensor.setBackend(new CpuBackend());
        }
    }
}
