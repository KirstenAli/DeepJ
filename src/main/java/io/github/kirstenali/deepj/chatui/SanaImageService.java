package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.diffusion.sana.SanaSnapshot;
import io.github.kirstenali.deepj.nn.diffusion.sana.SanaSprint;
import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.function.Consumer;

final class SanaImageService implements ImageService {
    @Override public Path resolve(Path selection) throws Exception { return SanaSnapshot.directory(selection); }

    @Override public BufferedImage generate(Path model, String prompt, SanaSprint.Options options,
                                            Consumer<String> progress) throws Exception {
        return new SanaSprint(model, progress).generate(prompt, options);
    }
}
