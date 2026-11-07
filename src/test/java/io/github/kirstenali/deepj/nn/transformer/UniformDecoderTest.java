package io.github.kirstenali.deepj.nn.transformer;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.nn.transformer.loop.DeepJLoop;
import io.github.kirstenali.deepj.nn.transformer.loop.DeepJLoopConfig;
import io.github.kirstenali.deepj.nn.transformer.prism.DeepJPrism;
import io.github.kirstenali.deepj.nn.transformer.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3;
import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3Config;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import io.github.kirstenali.deepj.training.CausalLMTraining;
import java.util.Random;
import java.util.Arrays;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

class UniformDecoderTest {
    @Test void compactMetalCachesPreserveFullForwardOutputsAcrossModelFamilies() {
        Assumptions.assumeTrue(MetalBackend.isAvailable());
        Tensor.setBackend(new MetalBackend());
        try {
            for (DType dtype : new DType[]{DType.FLOAT16, DType.BFLOAT16}) {
                for (String family : new String[]{"qwen", "loop", "prism"}) { cache(model(family, dtype), dtype); }
            }
        } finally {
            Tensor.backend().releaseResources();
            Tensor.setBackend(new CpuBackend());
        }
    }

    private static void cache(CausalLM model, DType dtype) {
        int[] tokens = new Random(3).ints(8, 0, 32).toArray();
        try (var ignored = GradMode.noGrad(); var session = model.openGeneration()) {
            for (int length = 4; length <= tokens.length; length++) {
                int[] prefix = Arrays.copyOf(tokens, length);
                Tensor cached = session.apply(prefix);
                assertEquals(dtype, cached.dtype());
                assertArrayEquals(model.forward(prefix).select(0, length - 1).toFloatArray(),
                        cached.toFloatArray(), dtype == DType.FLOAT16 ? .001f : .008f);
            }
        }
    }

    @ParameterizedTest @ValueSource(strings = {"cpu", "metal"})
    void compactModelsTrainThroughSharedTensorsAndPreserveFloat32Gradients(String backend) {
        if (backend.equals("metal")) Assumptions.assumeTrue(MetalBackend.isAvailable());
        Tensor.setBackend(backend.equals("metal") ? new MetalBackend() : new CpuBackend());
        try {
            for (DType dtype : new DType[]{DType.FLOAT16, DType.BFLOAT16}) {
                for (String family : new String[]{"qwen", "loop", "prism"}) { train(model(family, dtype), dtype); }
            }
        } finally {
            Tensor.backend().releaseResources();
            Tensor.setBackend(new CpuBackend());
        }
    }

    private static CausalLM model(String family, DType dtype) {
        var factory = ParameterFactory.uniform(dtype);
        return switch (family) {
            case "loop" -> new DeepJLoop(new DeepJLoopConfig(32, 32, 128, 4, 1, 0, 1, 0, 1, 256), 17, factory);
            case "prism" -> new DeepJPrism(new DeepJPrismConfig(32, 32, 128, 4, 1, 256, 32, 16), 17, factory);
            default -> new Qwen3(new Qwen3Config(32, 32, 128, 4, 2, 1, 256, 32, 1_000_000, .02f, 1), 17, factory);
        };
    }

    private static void train(CausalLM model, DType dtype) {
        int[] tokens = new Random(3).ints(16, 0, 32).toArray(), targets = new Random(4).ints(16, 0, 32).toArray();
        try (var ignored = GradMode.noGrad()) { assertEquals(dtype, model.forward(tokens).dtype()); }
        var batch = new Batch(new int[][]{tokens}, new int[][]{targets});
        var trainer = CausalLMTraining.trainer(model, size -> batch, 0.001f);
        float first = trainer.trainStep(1), last = first;
        for (int step = 0; step < 5; step++) { last = trainer.trainStep(1); }
        assertTrue(last < first, model.getClass().getSimpleName() + " " + dtype + ": " + first + " -> " + last);
        for (var parameter : model.parameters()) {
            assertEquals(dtype, parameter.value.dtype());
            assertEquals(DType.FLOAT32, parameter.grad().dtype());
            for (float value : parameter.grad().toFloatArray()) { assertTrue(Float.isFinite(value)); }
        }
    }
}
