package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.loss.CrossEntropyLoss;
import io.github.kirstenali.deepj.nn.transformer.DecoderOnlyModel;
import io.github.kirstenali.deepj.tensor.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import java.lang.reflect.Proxy;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;
import static io.github.kirstenali.deepj.testing.NdModelFixtures.*;
import static org.junit.jupiter.api.Assertions.*;

class MetalNdModelBatchTest extends MetalDifferentialTestSupport {
    static Stream<org.junit.jupiter.params.provider.Arguments> typed() { return typedCases(); }

    @ParameterizedTest @MethodSource("typed")
    void actualGpuSingleSequenceMatchesEverySingletonBatchValueAndGradient(String family, DType dtype) {
        io.github.kirstenali.deepj.testing.NdSingleSequenceChecks.exactSingletonBatch(family, dtype);
    }

    static Stream<String> names() { return families(); }

    @ParameterizedTest
    @MethodSource("names")
    void allModelBatchesAndMaskedGradientsMatchIndependentCpuSequences(String family) {
        Tensor.setBackend(cpu);
        DecoderOnlyModel reference = model(family, 91);
        Tensor first = reference.forward(inputs()[0]), second = reference.forward(inputs()[1]);
        CrossEntropyLoss.forward(first, targets()[0], masks()[0]).backward();
        CrossEntropyLoss.forward(second, targets()[1], masks()[1]).backward();
        List<Tensor> expected = copyGradients(reference.parameters());
        Tensor.setBackend(metal);
        DecoderOnlyModel actual = model(family, 91);
        Tensor output = actual.forwardBatch(inputs());
        assertArrayEquals(new int[]{2, 4, 19}, output.shape());
        CrossEntropyLoss.forward(output.select(0, 0), targets()[0], masks()[0])
                .add(CrossEntropyLoss.forward(output.select(0, 1), targets()[1], masks()[1])).backward();
        assertTensorClose(first, output.select(0, 0), 2e-3f, 3e-2f);
        assertTensorClose(second, output.select(0, 1), 2e-3f, 3e-2f);
        assertGradientListsClose(expected, copyGradients(actual.parameters()), 3e-3f, 5e-2f);
    }

    @ParameterizedTest
    @MethodSource("names")
    void ndBatchesDoNotDownloadIntermediatesOrAttendAcrossSequences(String family) {
        AtomicInteger downloads = new AtomicInteger();
        DecoderOnlyModel model = model(family, 92);
        Tensor.setBackend(counting(downloads));
        Tensor output = model.forwardBatch(inputs());
        output.sumTensor().backward();
        assertEquals(0, downloads.get());
        Tensor.setBackend(metal);
        int[][] changed = inputs();
        changed[0][3] = 18;
        changed[1] = new int[]{18, 17, 16, 15};
        try (var ignored = GradMode.noGrad()) {
            Tensor other = model.forwardBatch(changed);
            assertArrayEquals(output.select(0, 0).slice(0, 0, 3, 1).toFloatArray(),
                    other.select(0, 0).slice(0, 0, 3, 1).toFloatArray());
        }
    }

    private TensorBackend counting(AtomicInteger downloads) {
        return (TensorBackend) Proxy.newProxyInstance(TensorBackend.class.getClassLoader(),
                new Class<?>[]{TensorBackend.class}, (proxy, method, args) -> {
                    if (method.getName().equals("materializeTensor")) downloads.incrementAndGet();
                    return method.invoke(metal, args);
                });
    }
}
