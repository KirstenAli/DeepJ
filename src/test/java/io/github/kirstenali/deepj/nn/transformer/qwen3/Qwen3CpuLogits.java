package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/** Optional full-checkpoint parity probe: writes every last-position logit for three fixed contexts. */
public final class Qwen3CpuLogits {
    private Qwen3CpuLogits() {}

    public static void main(String[] args) throws Exception {
        Tensor.setBackend(new CpuBackend());
        DeepJExecutor.setNumThreads(10);
        Qwen3 model = Qwen3Pretrained.load(Path.of(args[0]), DType.valueOf(args[1]));
        var input = Qwen3Json.read(Path.of(args[2]));
        int[] prompt = ids(input.get("prompt")), decode = ids(input.get("decode"));
        try (var session = model.openGeneration();
                var stream = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(Path.of(args[3]))))) {
            int[] context = Arrays.copyOf(prompt, prompt.length + 2);
            for (int step = 0; step < 3; step++) {
                if (step > 0) context[prompt.length + step - 1] = decode[step - 1];
                for (float value : session.apply(Arrays.copyOf(context, prompt.length + step)).toFloatArray()) {
                    stream.writeFloat(value);
                }
            }
        } finally { DeepJExecutor.shutdown(); }
    }

    private static int[] ids(Object value) {
        return Qwen3Json.array(value).stream().mapToInt(token -> Math.toIntExact(Qwen3Json.integer(token))).toArray();
    }
}
