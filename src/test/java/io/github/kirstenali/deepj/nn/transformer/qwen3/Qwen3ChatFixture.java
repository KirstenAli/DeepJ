package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorEncoding;
import io.github.kirstenali.deepj.tokenizers.qwen3.Qwen3TokenizerFixture;
import java.io.IOException;
import java.nio.file.Path;

/** A complete tiny snapshot whose tied embedding deterministically generates printable x tokens. */
public final class Qwen3ChatFixture {
    private Qwen3ChatFixture() {}

    public static Path create(Path directory) throws IOException {
        var config = new Qwen3Config(263, 256, 12, 4, 2, 1, 20, 4, 1_000_000, 0.02f, 1);
        Qwen3 model = new Qwen3(config, 4);
        try (var ignored = GradMode.noGrad()) {
            for (var parameter : model.parameters()) {
                if (!parameter.name().endsWith(".gamma")) parameter.value.multiplyScalarInPlace(0);
            }
            float[] embedding = new float[263 * 12];
            embedding[10 * 12] = 1;
            embedding[120 * 12] = 2;
            TensorEncoding.restore(model.parameters().get(0).value, Tensor.from(embedding, 263, 12));
        }
        Qwen3Checkpoint.save(model, directory);
        Qwen3TokenizerFixture.write(directory);
        return directory;
    }
}
