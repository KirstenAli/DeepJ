package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.nn.ModelMemory;
import io.github.kirstenali.deepj.nn.ModelOps;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.Arrays;
import java.util.stream.IntStream;

/** Small guidance batches share one streamed weight pass; larger grids keep bounded sequential execution. */
final class VideoGuidance {
    private static final int BATCHED_LATENT_ELEMENTS = 65536;
    private VideoGuidance() {}

    static Tensor predict(SanaVideoTransformer model, Tensor latent, Tensor text, Tensor negative,
                          int[] valid, int[] negativeValid, float time, float guidance, DType dtype) {
        Tensor input = latent.to(dtype), caption = text.to(dtype);
        if (guidance > 1 && latent.numel() <= BATCHED_LATENT_ELEMENTS) {
            int[] mask = IntStream.concat(Arrays.stream(negativeValid), Arrays.stream(valid)).toArray();
            Tensor pair = model.forward(ModelOps.cat(0, input, input),
                    ModelOps.cat(0, negative.to(dtype), caption), mask, time).to(DType.FLOAT32);
            return combine(pair.slice(0, 1, 2), pair.slice(0, 0, 1), guidance);
        }
        Tensor positive = model.forward(input, caption, valid, time).to(DType.FLOAT32);
        if (guidance <= 1) return positive;
        try (var fixed = ModelMemory.retain(positive)) {
            return combine(positive, model.forward(input, negative.to(dtype), negativeValid, time)
                    .to(DType.FLOAT32), guidance);
        }
    }

    private static Tensor combine(Tensor positive, Tensor negative, float guidance) {
        return negative.add(positive.subtract(negative).multiplyScalar(guidance));
    }
}
