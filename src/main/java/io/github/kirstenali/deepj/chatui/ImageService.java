package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.diffusion.sana.SanaSprint;
import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.function.Consumer;

interface ImageService {
    Path resolve(Path selection) throws Exception;
    BufferedImage generate(Path model, String prompt, SanaSprint.Options options, Consumer<String> progress)
            throws Exception;
}
