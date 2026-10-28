package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.loss.CrossEntropyLoss;
import io.github.kirstenali.deepj.models.DecoderOnlyModel;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.GpuBuffer;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import io.github.kirstenali.deepj.testing.NdModelFixtures;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class MetalValidationOwnershipTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @MethodSource("io.github.kirstenali.deepj.testing.NdModelFixtures#typedCases")
    void scopedValidationReleasesLogitsWithoutCollectionOrUnloadingParameters(String family, DType dtype) {
        var model = NdModelFixtures.model(family, 417, new ParameterFactory(dtype));
        var retainedReferences = new ArrayList<Tensor>();
        Tensor caller = metal.addScalar(Tensor.ones(2, 2), 7).retainDeviceBuffer();
        try (var ignored = GradMode.noGrad()) {
            float expected = validation(model, retainedReferences);
            List<Object> parameterBuffers = parameterBuffers(model);
            Object callerBuffer = caller.getGpuTag();
            for (int round = 0; round < 40; round++) {
                assertEquals(Float.floatToRawIntBits(expected),
                        Float.floatToRawIntBits(validation(model, retainedReferences)));
                assertEquals(parameterBuffers, parameterBuffers(model), "Do not reload the model per validation");
                assertSame(callerBuffer, caller.getGpuTag());
            }
            assertArrayEquals(new float[]{8, 8, 8, 8}, caller.toFloatArray());
            assertEquals(41, retainedReferences.size(), "No reliance on collection of logit objects");
        } finally {
            caller.releaseDeviceBufferRetention();
        }
    }

    private float validation(DecoderOnlyModel model, List<Tensor> retainedReferences) {
        Tensor logits;
        int id;
        float loss;
        try (var ignored = TensorScope.open()) {
            logits = model.forward(new int[]{1, 2, 3, 4});
            retainedReferences.add(logits);
            loss = CrossEntropyLoss.loss(logits, new int[]{2, 3, 4, 5});
            id = ((GpuBuffer) logits.getGpuTag()).id;
            assertEquals(logits.storageBytes(), MetalNative.nativeBufferByteLength(id));
        }
        metal.releaseTemporaryResources();
        assertNull(logits.getGpuTag());
        assertThrows(RuntimeException.class, () -> MetalNative.nativeBufferByteLength(id),
                "The actual native allocation must be released even while the logit reference is alive");
        return loss;
    }

    private static List<Object> parameterBuffers(DecoderOnlyModel model) {
        var buffers = model.parameters().stream().map(parameter -> parameter.value.getGpuTag()).toList();
        assertTrue(buffers.stream().allMatch(GpuBuffer.class::isInstance));
        return buffers;
    }
}
