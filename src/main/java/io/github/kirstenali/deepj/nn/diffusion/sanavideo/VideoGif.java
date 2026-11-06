package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageTypeSpecifier;
import javax.imageio.ImageWriter;
import javax.imageio.metadata.IIOMetadata;
import javax.imageio.metadata.IIOMetadataNode;

/** Animated GIF export using Java's built-in writer; no Python or external video encoder is required. */
public final class VideoGif {
    private static final String FORMAT = "javax_imageio_gif_image_1.0";
    private VideoGif() {}

    public static void write(List<BufferedImage> frames, Path output, int fps) throws IOException {
        validate(frames, fps);
        var writers = ImageIO.getImageWritersByFormatName("gif");
        if (!writers.hasNext()) throw new IOException("Animated GIF writer unavailable");
        ImageWriter writer = writers.next();
        try (var stream = ImageIO.createImageOutputStream(output.toFile())) {
            writer.setOutput(stream);
            writer.prepareWriteSequence(null);
            for (int index = 0; index < frames.size(); index++) {
                int delay = (int) (Math.round((index + 1) * 100.0 / fps) - Math.round(index * 100.0 / fps));
                IIOMetadata metadata = metadata(writer, frames.get(index), delay, index == 0);
                writer.writeToSequence(new IIOImage(frames.get(index), null, metadata), null);
            }
            writer.endWriteSequence();
        } finally { writer.dispose(); }
    }

    private static void validate(List<BufferedImage> frames, int fps) {
        if (frames.isEmpty() || fps < 1 || fps > 100) throw new IllegalArgumentException("Invalid frames or FPS");
        int width = frames.get(0).getWidth(), height = frames.get(0).getHeight();
        for (BufferedImage frame : frames) {
            if (frame.getWidth() != width || frame.getHeight() != height) {
                throw new IllegalArgumentException("Video frames must have the same dimensions");
            }
        }
    }

    private static IIOMetadata metadata(ImageWriter writer, BufferedImage frame, int delay, boolean first)
            throws IOException {
        var type = ImageTypeSpecifier.createFromRenderedImage(frame);
        IIOMetadata metadata = writer.getDefaultImageMetadata(type, writer.getDefaultWriteParam());
        var tree = (IIOMetadataNode) metadata.getAsTree(FORMAT);
        var control = (IIOMetadataNode) tree.getElementsByTagName("GraphicControlExtension").item(0);
        control.setAttribute("disposalMethod", "none");
        control.setAttribute("userInputFlag", "FALSE");
        control.setAttribute("transparentColorFlag", "FALSE");
        control.setAttribute("delayTime", Integer.toString(delay));
        control.setAttribute("transparentColorIndex", "0");
        if (first) loop(tree);
        metadata.setFromTree(FORMAT, tree);
        return metadata;
    }

    private static void loop(IIOMetadataNode root) {
        var extensions = new IIOMetadataNode("ApplicationExtensions");
        var loop = new IIOMetadataNode("ApplicationExtension");
        loop.setAttribute("applicationID", "NETSCAPE");
        loop.setAttribute("authenticationCode", "2.0");
        loop.setUserObject(new byte[]{1, 0, 0});
        extensions.appendChild(loop);
        root.appendChild(extensions);
    }
}
