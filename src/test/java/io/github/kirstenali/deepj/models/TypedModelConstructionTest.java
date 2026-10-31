package io.github.kirstenali.deepj.models;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.loss.CrossEntropyLoss;
import io.github.kirstenali.deepj.optimisers.*;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import static io.github.kirstenali.deepj.testing.NdModelFixtures.*;
import static org.junit.jupiter.api.Assertions.*;

class TypedModelConstructionTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest
    @MethodSource("io.github.kirstenali.deepj.testing.NdModelFixtures#typedCases")
    void everyModelHasNamedTypedParametersAndFp32BatchedGradients(String family, DType dtype) {
        DecoderOnlyModel model = model(family, 95, new ParameterFactory(dtype));
        Tensor logits = model.forwardBatch(inputs());
        assertArrayEquals(new int[]{2, 4, 19}, logits.shape());
        assertEquals(dtype, logits.dtype());
        CrossEntropyLoss.forward(logits.select(0, 0), targets()[0], masks()[0])
                .add(CrossEntropyLoss.forward(logits.select(0, 1), targets()[1], masks()[1])).backward();
        var names = new HashSet<String>();
        for (Parameter parameter : model.parameters()) {
            assertTrue(names.add(parameter.name()), "Repeated parameter identity or name: " + parameter.name());
            assertParameter(dtype, parameter);
        }
        if (family.equals("origin")) assertEquals("positions.weight", model.parameters().get(1).name());
    }

    private static void assertParameter(DType dtype, Parameter parameter) {
        assertFalse(parameter.name().isEmpty());
        assertEquals(dtype, parameter.value.dtype());
        assertEquals(dtype.storageBytes(parameter.value.numel()), parameter.value.storageBytes());
        assertEquals(DType.FLOAT32, parameter.grad().dtype());
        for (float value : parameter.grad().toFloatArray()) { assertTrue(Float.isFinite(value)); }
    }

    @ParameterizedTest
    @MethodSource("io.github.kirstenali.deepj.testing.NdModelFixtures#families")
    void namedMixedDtypesDoNotDependOnConstructorOrParameterListingOrder(String family) {
        DecoderOnlyModel reference = model(family, 96);
        Map<String, TensorDescriptor> descriptors = mixedDescriptors(reference);
        DecoderOnlyModel mixed = model(family, 96, new ParameterFactory(descriptors));
        assertEquals(reference.parameters().size(), mixed.parameters().size());
        for (Parameter parameter : mixed.parameters()) {
            assertEquals(descriptors.get(parameter.name()), new TensorDescriptor(
                    parameter.value.dtype(), parameter.value.shape()));
        }
        assertFinite(mixed.forward(inputs()[0]));
        assertArrayEquals(reference.forward(inputs()[0]).toFloatArray(),
                model(family, 96, ParameterFactory.float32()).forward(inputs()[0]).toFloatArray());
    }

    private static Map<String, TensorDescriptor> mixedDescriptors(DecoderOnlyModel model) {
        Map<String, TensorDescriptor> result = new HashMap<>();
        int index = 0;
        for (Parameter parameter : model.parameters()) {
            result.put(parameter.name(), new TensorDescriptor(DType.values()[index++ % 3], parameter.value.shape()));
        }
        return result;
    }

    @ParameterizedTest
    @MethodSource("io.github.kirstenali.deepj.testing.NdModelFixtures#typedCases")
    void compactModelsUseIncrementalSessionsWithoutChangingWithinWindowResults(String family, DType dtype) {
        DecoderOnlyModel model = model(family, 97, new ParameterFactory(dtype));
        try (var ignored = GradMode.noGrad(); var session = model.openGeneration()) {
            for (int length = 1; length <= 4; length++) {
                int[] prefix = java.util.Arrays.copyOf(inputs()[0], length);
                Tensor expected = model.forward(prefix).slice(0, length - 1, length);
                Tensor actual = session.apply(prefix);
                assertEquals(dtype, actual.dtype());
                assertArrayEquals(expected.toFloatArray(), actual.toFloatArray(), 3e-6f);
            }
        }
    }

    private static void assertFinite(Tensor tensor) {
        for (float value : tensor.toFloatArray()) { assertTrue(Float.isFinite(value)); }
    }
}
