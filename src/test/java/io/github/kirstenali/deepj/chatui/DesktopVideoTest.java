package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.diffusion.sanavideo.SanaVideo;
import io.github.kirstenali.deepj.nn.diffusion.sanavideo.VideoMp4;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.IntStream;
import org.jcodec.api.FrameGrab;
import org.jcodec.common.io.NIOUtils;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class DesktopVideoTest {
    @TempDir Path temporary;

    @Test void mp4PreservesFrameCountDimensionsAndPlaybackTiming() throws Exception {
        var request = new VideoRequest(new SanaVideo.Options(128, 96, 9, 1, 6, 42), 16);
        Path folder;
        try (var video = DesktopVideo.encode(frames(), request);
             var channel = NIOUtils.readableChannel(video.mp4().toFile())) {
            folder = video.directory();
            var movie = FrameGrab.createFrameGrab(channel);
            assertEquals(9, movie.getVideoTrack().getMeta().getTotalFrames());
            assertEquals(9.0 / 16, movie.getVideoTrack().getMeta().getTotalDuration(), 1e-6);
            assertEquals(128, movie.getNativeFrame().getWidth());
            movie.seekToFramePrecise(8);
            assertEquals(96, movie.getNativeFrame().getHeight());
            assertTrue(Files.isRegularFile(video.poster()));
            assertEquals("deepj-video-42.mp4", video.filename());
        }
        assertFalse(Files.exists(folder));
    }

    @Test void invalidOrInterruptedExportsDoNotOverwriteExistingFiles() throws Exception {
        Path path = temporary.resolve("keep.mp4");
        Files.writeString(path, "keep");
        assertThrows(IllegalArgumentException.class, () -> VideoMp4.write(List.of(), path, 16));
        assertThrows(IllegalArgumentException.class, () -> VideoMp4.write(frames(), path, 0));
        Thread.currentThread().interrupt();
        try {
            assertThrows(java.util.concurrent.CancellationException.class, () -> VideoMp4.write(frames(), path, 16));
        }
        finally { Thread.interrupted(); }
        assertEquals("keep", Files.readString(path));
        try (var files = Files.list(temporary)) { assertEquals(1, files.count()); }
    }

    static List<BufferedImage> frames() { return IntStream.range(0, 9).mapToObj(DesktopVideoTest::frame).toList(); }

    private static BufferedImage frame(int index) {
        var image = new BufferedImage(128, 96, BufferedImage.TYPE_INT_RGB);
        var graphics = image.createGraphics();
        graphics.setColor(new java.awt.Color(12, 28, 44));
        graphics.fillRect(0, 0, 128, 96);
        graphics.setColor(new java.awt.Color(46, 185, 235));
        graphics.fillOval(8 + index * 10, 32, 24, 24);
        graphics.dispose();
        return image;
    }
}
