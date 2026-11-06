package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class SanaVideoPipelineTest {
    private static final Path MODEL = Path.of("src/test/resources/sana-video/pipeline");
    @TempDir Path output;

    @BeforeEach void backend() {
        Tensor.setBackend(Boolean.getBoolean("sanavideo.test.metal") ? new MetalBackend() : new CpuBackend());
    }

    @Test void stagedPipelineGeneratesFiveFramesAndReleasesHistoryBetweenVideos() throws Exception {
        List<String> progress = new ArrayList<>();
        var pipeline = new SanaVideo(MODEL, progress::add);
        var options = new SanaVideo.Options(16, 16, 5, 2, 6, 123);
        List<BufferedImage> first = pipeline.generate("a fox", options);
        List<BufferedImage> repeat = pipeline.generate("a fox", options);
        assertEquals(5, first.size());
        assertEquals(16, first.get(0).getWidth());
        assertTrue(progress.stream().anyMatch(message -> message.contains("step 2/2")));
        assertTrue(GradMode.isEnabled());
        for (int frame = 0; frame < 5; frame++) {
            assertArrayEquals(first.get(frame).getRGB(0, 0, 16, 16, null, 0, 16),
                    repeat.get(frame).getRGB(0, 0, 16, 16, null, 0, 16));
        }
    }

    @Test void cliExportsAnAnimatedGifWithAllFramesAndRequestedTiming() throws Exception {
        Path gif = output.resolve("video.gif");
        int status = SanaVideoCli.command().execute("--model", MODEL.toString(), "--backend", "cpu",
                "--width", "16", "--height", "16", "--frames", "5", "--steps", "1", "--guidance", "1",
                "--output", gif.toString(), "a fox");
        assertEquals(0, status);
        var reader = ImageIO.getImageReadersByFormatName("gif").next();
        try (var input = ImageIO.createImageInputStream(gif.toFile())) {
            reader.setInput(input);
            assertEquals(5, reader.getNumImages(true));
            assertEquals(16, reader.read(4).getWidth());
            int duration = 0;
            for (int i = 0; i < 5; i++) { duration += delay(reader.getImageMetadata(i)); }
            assertEquals(31, duration);
        } finally { reader.dispose(); }
    }

    private static int delay(javax.imageio.metadata.IIOMetadata metadata) {
        var root = (javax.imageio.metadata.IIOMetadataNode) metadata.getAsTree("javax_imageio_gif_image_1.0");
        var control = root.getElementsByTagName("GraphicControlExtension").item(0);
        return Integer.parseInt(control.getAttributes().getNamedItem("delayTime").getNodeValue());
    }

    @Test void rgbUsesTheSameHalfwayRoundingAsDiffusers() {
        Tensor pixels = Tensor.from(new float[]{0.4f, -0.4f, 0}, 1, 3, 1, 1, 1);
        int rgb = VideoFrames.rgb(pixels).get(0).getRGB(0, 0) & 0xffffff;
        assertEquals((178 << 16) | (76 << 8) | 128, rgb);
    }

    @Test void cancellationRestoresTheCallersGradientMode() throws Exception {
        var restored = new AtomicBoolean();
        Thread worker = new Thread(() -> {
            Thread.currentThread().interrupt();
            try { new SanaVideo(MODEL).generate("fox", new SanaVideo.Options(16, 16, 5, 1, 1, 1)); }
            catch (CancellationException expected) { restored.set(GradMode.isEnabled()); }
            catch (java.io.IOException failure) { throw new AssertionError(failure); }
        });
        worker.start();
        worker.join(5000);
        assertFalse(worker.isAlive());
        assertTrue(restored.get());
    }

    @Test void invalidVideoDimensionsAndExportsFailBeforeInference() {
        assertThrows(IllegalArgumentException.class, () -> new SanaVideo.Options(15, 16, 5, 1, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> new SanaVideo.Options(16, 16, 8, 1, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> new SanaVideo.Options(16, 16, 5, 0, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> new FlowDpmSolver(5, Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> VideoGif.write(List.of(), output.resolve("a.gif"), 16));
        assertThrows(IllegalArgumentException.class, () -> VideoFrames.rgb(Tensor.zeros(1, 3)));
    }

}
