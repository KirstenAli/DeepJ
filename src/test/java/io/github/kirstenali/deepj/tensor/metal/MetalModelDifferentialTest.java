package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.data.BatchSource;
import io.github.kirstenali.deepj.models.DecoderOnlyModel;
import io.github.kirstenali.deepj.models.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.models.prism.DeepJPrism;
import io.github.kirstenali.deepj.models.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.models.origin.DeepJOrigin;
import io.github.kirstenali.deepj.models.orbit.DeepJOrbitConfig;
import io.github.kirstenali.deepj.models.orbit.DeepJOrbit;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.CrossEntropyResult;
import io.github.kirstenali.deepj.tensor.RmsNormResult;
import io.github.kirstenali.deepj.tensor.SwiGluBackwardResult;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.training.CausalLMTraining;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class MetalModelDifferentialTest extends MetalDifferentialTestSupport {
    @Test
    void originForwardBackwardAndParameterGradientsMatchCpu() {
        compareModel(() -> new DeepJOrigin(new DeepJOriginConfig(11, 4, 4, 2, 1, 6), 21L));
    }

    @Test
    void orbitForwardBackwardAndParameterGradientsMatchCpu() {
        compareModel(() -> new DeepJOrbit(new DeepJOrbitConfig(11, 4, 4, 2, 1, 8), 22L));
    }

    @Test
    void prismForwardBackwardAndParameterGradientsMatchCpu() {
        compareModel(() -> new DeepJPrism(new DeepJPrismConfig(11, 4, 4, 2, 1, 8, 3, 2), 23L));
    }

    @Test
    void prismForwardBackwardDoesNotDownloadIntermediateTensors() {
        AtomicInteger downloads = new AtomicInteger();
        var config = new DeepJPrismConfig(11, 4, 4, 2, 1, 8, 3, 2);
        DecoderOnlyModel model = new DeepJPrism(config, 25L);
        Tensor.setBackend(countingBackend(downloads));
        model.backward(model.forward(new int[]{1, 3, 5}).multiplyScalar(0.25f));
        assertEquals(0, downloads.get());
    }

    private TensorBackend countingBackend(AtomicInteger downloads) {
        return (TensorBackend) Proxy.newProxyInstance(
                TensorBackend.class.getClassLoader(), new Class<?>[]{TensorBackend.class},
                (proxy, method, args) -> invokeMetal(method.getName(), method, args, downloads));
    }

    private Object invokeMetal(String name, java.lang.reflect.Method method,
                               Object[] args, AtomicInteger downloads) throws Exception {
        if (name.equals("materializeTensor")) downloads.incrementAndGet();
        return method.invoke(metal, args);
    }

    @Test
    void temporaryReleaseDoesNotChangeRepeatedPrismTraining() {
        MetalBackend baseline = new MetalBackend();
        try {
            assertRepeatedTrainingMatches(baseline);
        } finally {
            baseline.releaseResources();
        }
    }

    private void assertRepeatedTrainingMatches(MetalBackend baseline) {
        var config = new DeepJPrismConfig(11, 4, 4, 2, 1, 8, 3, 2);
        DecoderOnlyModel expected = modelOn(baseline, config);
        DecoderOnlyModel actual = modelOn(metal, config);
        assertEquals(trainRepeated(baseline, expected, 0), trainRepeated(metal, actual, 1), 1e-5f);
        assertParameterValuesClose(expected.parameters(), actual.parameters(), 1e-5f);
    }

    private static DecoderOnlyModel modelOn(TensorBackend backend, DeepJPrismConfig config) {
        Tensor.setBackend(backend);
        return new DeepJPrism(config, 24L);
    }

    private static float trainRepeated(TensorBackend backend, DecoderOnlyModel model,
                                       int releaseEvery) {
        Tensor.setBackend(backend);
        BatchSource source = ignored -> new Batch(new int[][]{{1, 3, 5, 7}},
                new int[][]{{3, 5, 7, 2}});
        return CausalLMTraining.trainer(model, source, 1e-3f)
                .train(3, 1, 1000, 0.98f, null, releaseEvery).lastLoss();
    }

    private static void assertParameterValuesClose(List<Parameter> expected,
                                                   List<Parameter> actual, float tolerance) {
        assertEquals(expected.size(), actual.size());
        for (int index = 0; index < expected.size(); index++) {
            assertTensorClose(expected.get(index).value, actual.get(index).value,
                    tolerance, tolerance, "parameter " + index);
        }
    }

    private void compareModel(Supplier<DecoderOnlyModel> factory) {
        Tensor.setBackend(cpu);
        DecoderOnlyModel expectedModel = factory.get();
        DecoderOnlyModel actualModel = factory.get();
        ModelResult expected = runModel(expectedModel, new int[]{1, 3, 5}, 30L);
        Tensor.setBackend(metal);
        ModelResult actual = runModel(actualModel, new int[]{1, 3, 5}, 30L);
        assertTensorClose(expected.output(), actual.output(), 2e-3f, 3e-2f);
        assertGradientListsClose(expected.gradients(), actual.gradients(), 3e-3f, 5e-2f);
    }

    private static ModelResult runModel(DecoderOnlyModel model, int[] inputIds, long seed) {
        model.zeroGrad();
        Tensor output = model.forward(inputIds);
        Tensor upstream = Tensor.random(output.rows, output.cols, new Random(seed));
        model.backward(upstream);
        return new ModelResult(output, copyGradients(model.parameters()));
    }


}
