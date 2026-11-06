package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.tensor.Tensor;

/** Layout conversions for channel-first [batch, channels, time, height, width] video tensors. */
final class VideoLayout {
    private VideoLayout() {}

    static Tensor tokens(Tensor video) {
        return video.permute(0, 2, 3, 4, 1).reshape(video.size(0),
                video.size(2) * video.size(3) * video.size(4), video.size(1));
    }

    static Tensor volume(Tensor tokens, int time, int height, int width) {
        return tokens.reshape(tokens.size(0), time, height, width, tokens.size(2)).permute(0, 4, 1, 2, 3);
    }

    static Tensor frames(Tensor video) {
        return video.permute(0, 2, 1, 3, 4)
                .reshape(video.size(0) * video.size(2), video.size(1), video.size(3), video.size(4));
    }

    static Tensor video(Tensor frames, int batch, int time) {
        return frames.reshape(batch, time, frames.size(1), frames.size(2), frames.size(3))
                .permute(0, 2, 1, 3, 4);
    }
}
