package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;

/** Optional, synchronized inference benchmark; never runs as part of the unit-test suite. */
public final class Qwen3Benchmark {
    private Qwen3Benchmark() {}

    public static void main(String[] args) throws Exception {
        boolean metal = args[1].equals("metal");
        if (metal && !MetalBackend.isAvailable()) throw new IllegalStateException("Physical Metal required");
        Tensor.setBackend(metal ? new MetalBackend() : new CpuBackend());
        DeepJExecutor.setNumThreads(args.length > 4 ? Integer.parseInt(args[4]) : 10);
        boolean compact = args.length > 7 && Boolean.parseBoolean(args[7]);
        Qwen3 model = Qwen3Pretrained.load(Path.of(args[0]), DType.valueOf(args[2]), compact);
        try { measure(model, args); }
        finally {
            // This standalone process exits next; do not download the entire GPU model during teardown.
            Tensor.backend().releaseTemporaryResources();
            DeepJExecutor.shutdown();
        }
    }

    private static void measure(Qwen3 model, String[] args) throws Exception {
        var input = Qwen3Json.read(Path.of(args[3]));
        int warmups = args.length > 5 ? Integer.parseInt(args[5]) : 1;
        int runs = args.length > 6 ? Integer.parseInt(args[6]) : 3;
        for (int iteration = -warmups; iteration < runs; iteration++) {
            trial(model, ids(input.get("prompt")), ids(input.get("decode")), iteration, args[1], args[2]);
        }
    }

    private static int[] ids(Object value) {
        return Qwen3Json.array(value).stream().mapToInt(token -> Math.toIntExact(Qwen3Json.integer(token))).toArray();
    }

    private static void trial(Qwen3 model, int[] prompt, int[] decode, int iteration, String backend, String dtype) {
        try (var session = model.openGeneration()) {
            long start = System.nanoTime();
            float check = session.apply(prompt).toFloatArray()[0];
            double prefill = (System.nanoTime() - start) / 1e9;
            int[] context = Arrays.copyOf(prompt, prompt.length + decode.length);
            start = System.nanoTime();
            for (int index = 0; index < decode.length; index++) {
                context[prompt.length + index] = decode[index];
                check += session.apply(Arrays.copyOf(context, prompt.length + index + 1)).toFloatArray()[0];
                Tensor.backend().releaseTemporaryResources();
            }
            double seconds = (System.nanoTime() - start) / 1e9;
            System.out.println(Qwen3Json.stringify(Map.of("engine", "DeepJ", "backend", backend, "dtype", dtype,
                    "trial", iteration, "prefill_seconds", prefill, "decode_seconds", seconds,
                    "tokens_per_second", decode.length / seconds, "checksum", check)));
        }
    }
}
