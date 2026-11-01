package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.loss.CrossEntropyLoss;
import io.github.kirstenali.deepj.nn.transformer.DecoderOnlyModel;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.testing.TypedParameterPolicy;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import static io.github.kirstenali.deepj.testing.NdModelFixtures.*;
import static org.junit.jupiter.api.Assertions.*;

class MetalTypedModelConstructionTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @MethodSource("io.github.kirstenali.deepj.testing.NdModelFixtures#typedCases")
    void allFamiliesDifferentiateTypedBatchesOnDeviceAgainstCpu(String family, DType dtype) {
        Tensor.setBackend(cpu);
        ModelResult expected = run(model(family, 98, new ParameterFactory(dtype)));
        Tensor.setBackend(metal);
        DecoderOnlyModel actual = model(family, 98, new ParameterFactory(dtype));
        AtomicInteger downloads = new AtomicInteger();
        Tensor.setBackend(counting(downloads));
        ModelResult result = run(actual);
        assertEquals(0, downloads.get());
        assertFalse(TensorTestStorage.hasHostStorage(result.output()));
        Tensor.setBackend(metal);
        assertEquals(DType.FLOAT32, result.output().dtype());
        assertTensorClose(expected.output(), result.output(), 2e-3f, 3e-2f);
        assertGradientListsClose(expected.gradients(), result.gradients(), 3e-3f, 5e-2f);
        assertParameters(actual, dtype);
    }

    private static ModelResult run(DecoderOnlyModel model) {
        Tensor output = model.forwardBatch(inputs());
        CrossEntropyLoss.forward(output.select(0, 0), targets()[0], masks()[0])
                .add(CrossEntropyLoss.forward(output.select(0, 1), targets()[1], masks()[1])).backward();
        return new ModelResult(output, model.parameters().stream().map(parameter -> parameter.grad()).toList());
    }

    private static void assertParameters(DecoderOnlyModel model, DType dtype) {
        for (var parameter : model.parameters()) {
            TypedParameterPolicy.assertStorage(dtype, parameter);
        }
    }

    private TensorBackend counting(AtomicInteger downloads) {
        return (TensorBackend) Proxy.newProxyInstance(TensorBackend.class.getClassLoader(),
                new Class<?>[]{TensorBackend.class}, (proxy, method, args) -> {
                    if (method.getName().equals("materializeTensor")) downloads.incrementAndGet();
                    return method.invoke(metal, args);
                });
    }

    @ParameterizedTest
    @MethodSource("io.github.kirstenali.deepj.testing.NdModelFixtures#typedCases")
    void allFamiliesUseTypedIncrementalSessionsOnDevice(String family, DType dtype) {
        Tensor.setBackend(cpu);
        List<Tensor> expected = prefixes(model(family, 99, new ParameterFactory(dtype)));
        Tensor.setBackend(metal);
        List<Tensor> actual = prefixes(model(family, 99, new ParameterFactory(dtype)));
        try {
            for (int index = 0; index < expected.size(); index++) {
                assertEquals(DType.FLOAT32, actual.get(index).dtype());
                assertTensorClose(expected.get(index), actual.get(index), 2e-3f, 3e-2f);
            }
        } finally {
            expected.forEach(Tensor::releaseDeviceBufferRetention);
            actual.forEach(Tensor::releaseDeviceBufferRetention);
        }
    }

    private static List<Tensor> prefixes(DecoderOnlyModel model) {
        List<Tensor> result = new ArrayList<>();
        try (var ignored = GradMode.noGrad(); var session = model.openGeneration()) {
            for (int length = 1; length <= 4; length++) {
                Tensor logits = session.apply(Arrays.copyOf(inputs()[0], length));
                result.add(logits.detach().retainDeviceBuffer());
            }
        }
        return result;
    }
}
