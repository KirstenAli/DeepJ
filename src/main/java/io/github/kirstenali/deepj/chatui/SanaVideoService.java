package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.diffusion.sanavideo.SanaVideo;
import io.github.kirstenali.deepj.nn.diffusion.sanavideo.SanaVideoSnapshot;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.List;
import java.util.function.Consumer;

final class SanaVideoService implements VideoService {
    @Override public Path resolve(Path path) throws Exception { return SanaVideoSnapshot.directory(path); }

    @Override public List<BufferedImage> generate(Path path, String prompt, SanaVideo.Options options,
                                                  Consumer<String> progress) throws Exception {
        return VideoWorkerProcess.generate(path, prompt, options, Tensor.backend() instanceof MetalBackend, progress);
    }
}
