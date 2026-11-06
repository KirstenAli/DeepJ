// Portions adapted from Diffusers (Apache-2.0); see META-INF/licenses/sana-video-NOTICE.txt.
package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.tensor.Tensor;

/** Adjacent-pair rotary embeddings with separate time, height, and width frequencies. */
final class VideoRotary {
    private final Tensor cosine;
    private final Tensor sine;
    private final int[] partners;
    private final Tensor signs;

    VideoRotary(int time, int height, int width, int channels) {
        if (channels < 6 || channels % 2 != 0) throw new IllegalArgumentException("Invalid rotary width");
        int spatial = 2 * (channels / 6);
        float[] cos = new float[time * height * width * channels], sin = new float[cos.length];
        for (int index = 0; index < cos.length; index++) {
            double angle = angle(index, channels, spatial, height, width);
            cos[index] = (float) Math.cos(angle);
            sin[index] = (float) Math.sin(angle);
        }
        cosine = Tensor.from(cos, 1, 1, time * height * width, channels);
        sine = Tensor.from(sin, cosine.shape());
        partners = java.util.stream.IntStream.range(0, channels).map(i -> i ^ 1).toArray();
        float[] sign = new float[channels];
        for (int i = 0; i < channels; i++) { sign[i] = i % 2 == 0 ? -1 : 1; }
        signs = Tensor.from(sign, channels);
    }

    private static double angle(int index, int channels, int spatial, int height, int width) {
        int channel = index % channels, token = index / channels, temporal = channels - 2 * spatial;
        int position = token / (height * width), local = channel, dimension = temporal;
        if (channel >= temporal) {
            local = (channel - temporal) % spatial;
            dimension = spatial;
            position = channel < temporal + spatial ? token / width % height : token % width;
        }
        return position * Math.pow(10000, -(2.0 * (local / 2)) / dimension);
    }

    Tensor apply(Tensor input) {
        return input.multiply(cosine).add(input.indexSelect(-1, partners).multiply(signs).multiply(sine));
    }

    Tensor[] buffers() { return new Tensor[]{cosine, sine, signs}; }
}
