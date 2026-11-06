package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.nn.ModelMemory;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import org.jcodec.api.awt.AWTSequenceEncoder;
import org.jcodec.common.io.NIOUtils;
import org.jcodec.common.model.Rational;

/** H.264 MP4 encoding in Java, independent of Python and external encoder executables. */
public final class VideoMp4 {
    private VideoMp4() {}

    public static void write(List<BufferedImage> frames, Path output, int fps) throws IOException {
        validate(frames, fps);
        ModelMemory.checkCancelled();
        Path target = output.toAbsolutePath();
        Path temporary = Files.createTempFile(target.getParent(), ".deepj-video-", ".mp4");
        try {
            encode(frames, temporary, fps);
            Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING);
        } finally { Files.deleteIfExists(temporary); }
    }

    private static void encode(List<BufferedImage> frames, Path output, int fps) throws IOException {
        try (var channel = NIOUtils.writableChannel(output.toFile())) {
            var encoder = new AWTSequenceEncoder(channel, Rational.R(fps, 1));
            for (BufferedImage frame : frames) {
                ModelMemory.checkCancelled();
                encoder.encodeImage(frame);
            }
            encoder.finish();
        }
    }

    private static void validate(List<BufferedImage> frames, int fps) {
        if (frames.isEmpty() || fps < 1 || fps > 100) throw new IllegalArgumentException("Invalid frames or FPS");
        int width = frames.get(0).getWidth(), height = frames.get(0).getHeight();
        if (((width | height) & 1) != 0) throw new IllegalArgumentException("MP4 dimensions must be even");
        for (BufferedImage frame : frames) {
            if (frame.getWidth() != width || frame.getHeight() != height) {
                throw new IllegalArgumentException("MP4 frames must have the same dimensions");
            }
        }
    }
}
