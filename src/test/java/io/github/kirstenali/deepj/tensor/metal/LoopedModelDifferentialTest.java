package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.nn.transformer.loop.DeepJLoop;
import io.github.kirstenali.deepj.nn.transformer.loop.DeepJLoopConfig;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.List;

class LoopedModelDifferentialTest extends MetalDifferentialTestSupport {

    @ParameterizedTest
    @CsvSource({"7,1", "7,2", "7,3", "256,2", "1024,2"})
    void outputsGradientsAndOptimizerUpdateMatchCpu(int length, int passes) {
        var config = new DeepJLoopConfig(67, length, 64, 1, 1, 0, 1, 0, passes, 96);
        int[] sequence = randomTargets(length, config.vocabSize(), 41L);
        Tensor upstream = random(length, config.vocabSize(), 43L);
        Snapshot expected = execute(config, sequence, upstream, cpu);
        Snapshot observed = execute(config, sequence, upstream, metal);
        assertTensorClose(expected.logits(), observed.logits(), 3e-4f, 3e-3f);
        assertGradientListsClose(expected.gradients(), observed.gradients(), 5e-4f, 5e-3f);
        assertGradientListsClose(expected.weights(), observed.weights(), 2e-4f, 2e-3f);
    }

    private static Snapshot execute(DeepJLoopConfig config, int[] sequence, Tensor upstream, TensorBackend backend) {
        Tensor.setBackend(backend);
        var decoder = new DeepJLoop(config, 47L);
        Tensor logits = decoder.forward(sequence);
        Tensor copied = new Tensor(logits);
        logits.backward(new Tensor(upstream));
        List<Tensor> gradients = copyGradients(decoder.parameters());
        AdamW.defaultAdamW(0.0001f).step(decoder.parameters());
        List<Tensor> weights = decoder.parameters().stream().map(p -> new Tensor(p.value)).toList();
        backend.releaseResources();
        return new Snapshot(copied, gradients, weights);
    }

    private record Snapshot(Tensor logits, List<Tensor> gradients, List<Tensor> weights) {}
}
