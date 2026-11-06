package io.github.kirstenali.deepj.cli;

import io.github.kirstenali.deepj.nn.diffusion.sanavideo.SanaVideoCli;
import java.nio.file.Files;
import java.nio.file.Path;
import org.jcodec.api.FrameGrab;
import org.jcodec.common.io.NIOUtils;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class VideoCliTest {
    @Test void mainCliExposesVideoHelp() {
        assertEquals(0, DeepJCli.execute(CliServices.system(), "video", "--help"));
        assertEquals(0, SanaVideoCli.command().execute("--help"));
        assertFalse(SanaVideoCli.command().getUsageMessage().contains("--dtype"));
        assertThrows(picocli.CommandLine.ParameterException.class,
                () -> SanaVideoCli.command().parseArgs("--model", "model", "--dtype", "bfloat16", "fox"));
    }

    @Test void actualVideoCommandGeneratesAndSavesMp4FromTheSmallPipeline(@TempDir Path directory) throws Exception {
        Path output = directory.resolve("generated.mp4");
        int status = DeepJCli.execute(CliServices.system(), "video",
                "--model", "src/test/resources/sana-video/pipeline",
                "--output", output.toString(), "--backend", "cpu", "--width", "16", "--height", "16",
                "--frames", "5", "--steps", "1", "A fox");
        assertEquals(0, status);
        assertTrue(Files.size(output) > 0);
        try (var channel = NIOUtils.readableChannel(output.toFile())) {
            var movie = FrameGrab.createFrameGrab(channel);
            assertEquals(5, movie.getVideoTrack().getMeta().getTotalFrames());
            assertNotNull(movie.getNativeFrame());
        }
    }
}
