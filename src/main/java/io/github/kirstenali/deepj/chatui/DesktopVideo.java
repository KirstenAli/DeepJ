package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.diffusion.sanavideo.VideoMp4;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.ArrayList;
import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import javax.imageio.ImageIO;

/** Stores MP4 and preview frames on disk instead of copying image bytes through the JavaScript bridge. */
record DesktopVideo(Path directory, List<String> frames, VideoRequest request) implements AutoCloseable {
    static DesktopVideo encode(List<BufferedImage> images, VideoRequest request) throws IOException {
        var result = new DesktopVideo(Files.createTempDirectory("deepj-video-"), new ArrayList<>(), request);
        try {
            VideoMp4.write(images, result.mp4(), request.fps());
            result.writeFrames(images);
            return result;
        } catch (IOException | RuntimeException failure) {
            result.close();
            throw failure;
        }
    }

    private void writeFrames(List<BufferedImage> images) throws IOException {
        for (int index = 0; index < images.size(); index++) {
            io.github.kirstenali.deepj.nn.ModelMemory.checkCancelled();
            Path frame = directory.resolve("frame-" + index + ".png");
            if (!ImageIO.write(images.get(index), "png", frame.toFile())) {
                throw new IOException("PNG writer unavailable");
            }
            frames.add(frame.toUri().toString());
        }
    }

    String frameUrls() { return Qwen3Json.stringify(frames); }
    Path mp4() { return directory.resolve("video.mp4"); }
    Path poster() { return directory.resolve("frame-0.png"); }
    String filename() { return "deepj-video-" + request.options().seed() + ".mp4"; }

    @Override public void close() {
        try {
            for (Path path : storedFiles()) { Files.deleteIfExists(path); }
            Files.deleteIfExists(directory);
        } catch (IOException failure) { directory.toFile().deleteOnExit(); }
    }

    private List<Path> storedFiles() throws IOException {
        try (var files = Files.list(directory)) {
            return files.toList();
        }
    }
}
