package io.github.kirstenali.deepj.nn.diffusion.sana;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.nio.file.Path;
import java.util.ArrayList;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import picocli.CommandLine;
import static org.junit.jupiter.api.Assertions.*;

class SanaPipelineTest {
    private static final Path MODEL = Path.of("src/test/resources/sana/pipeline");
    @TempDir Path temporary;

    @Test void stagedGenerationIsSeededAndPreservesGradMode() throws Exception {
        Tensor.setBackend(Boolean.getBoolean("sana.test.metal") ? new MetalBackend() : new CpuBackend());
        var stages = new ArrayList<String>();
        var pipeline = new SanaSprint(MODEL, stages::add);
        var options = new SanaSprint.Options(32, 32, 2, 4.5f, 7);
        var first = pipeline.generate("a fox", options);
        var second = pipeline.generate("a fox", options);
        assertEquals(32, first.getWidth());
        assertEquals(32, first.getHeight());
        assertArrayEquals(first.getRGB(0, 0, 32, 32, null, 0, 32), second.getRGB(0, 0, 32, 32, null, 0, 32));
        assertTrue(stages.contains("Denoising step 2/2"));
        assertTrue(io.github.kirstenali.deepj.autograd.GradMode.isEnabled());
        assertThrows(IllegalArgumentException.class, () -> pipeline.generate(" ", options));
    }

    @Test void cancellationBetweenStagesReleasesTheEncoderAndRestoresGradMode() {
        Tensor.setBackend(Boolean.getBoolean("sana.test.metal") ? new MetalBackend() : new CpuBackend());
        var pipeline = new SanaSprint(MODEL, message -> {
            if (message.startsWith("Loading")) Thread.currentThread().interrupt();
        });
        try {
            assertThrows(java.util.concurrent.CancellationException.class,
                    () -> pipeline.generate("a fox", new SanaSprint.Options(32, 32, 2, 4.5f, 1)));
        } finally { Thread.interrupted(); }
        assertTrue(io.github.kirstenali.deepj.autograd.GradMode.isEnabled());
    }

    @Test void standaloneLauncherWritesPngAndReportsInvalidOptions() throws Exception {
        Path output = temporary.resolve("image.png");
        int status = new CommandLine(new SanaImageCli()).execute("--model", MODEL.toString(), "--backend", "cpu",
                "--width", "32", "--height", "32", "--steps", "1", "--output", output.toString(), "a fox");
        assertEquals(0, status);
        assertEquals(32, ImageIO.read(output.toFile()).getWidth());
        assertNotEquals(0, new CommandLine(new SanaImageCli()).execute("--model", MODEL.toString(),
                "--backend", "invalid", "a fox"));
    }
}
