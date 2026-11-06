package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.diffusion.sanavideo.SanaVideo;
import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.List;
import java.util.function.Consumer;

interface VideoService {
    Path resolve(Path path) throws Exception;
    List<BufferedImage> generate(Path path, String prompt, SanaVideo.Options options, Consumer<String> progress)
            throws Exception;
}
