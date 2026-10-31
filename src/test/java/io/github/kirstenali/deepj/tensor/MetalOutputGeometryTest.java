package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.function.UnaryOperator;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class MetalOutputGeometryTest {
    private MetalBackend metal;

    @BeforeEach void device() {
        assumeTrue(MetalBackend.isAvailable(), "Metal device not available");
        metal = new MetalBackend();
    }
    @AfterEach void release() { if (metal != null) metal.releaseResources(); }

    @Test
    void unaryAndSoftmaxKernelsShareGeometryButNeverOutputState() {
        Tensor input = Tensor.ones(2, 4);
        List<UnaryOperator<Tensor>> operations = List.of(
                a -> metal.multiplyScalar(a, 2), a -> metal.addScalar(a, 1), a -> metal.divideScalar(a, 2),
                a -> metal.clamp(a, 0, 2), a -> metal.pow(a, 2), metal::sqrt, metal::neg, metal::exp,
                metal::log, metal::tanh, metal::sigmoid, metal::relu, metal::gelu, metal::softmaxRows,
                a -> metal.reluBackward(a, a), a -> metal.geluBackward(a, a),
                a -> metal.softmaxBackward(a, a), a -> metal.softmaxBackward(a, a, 0.5f));
        for (var operation : operations) { independent(input, operation.apply(input)); }
    }

    @Test
    void broadcastAndPositionOperationsKeepTheCorrectInputGeometry() {
        Tensor input = Tensor.ones(4, 4), row = Tensor.ones(1, 4), col = Tensor.ones(4, 1);
        List<UnaryOperator<Tensor>> operations = List.of(
                a -> metal.addRowVector(a, row), a -> metal.addBroadcastCols(a, col),
                a -> metal.divideBroadcastCols(a, col), a -> metal.subtractBroadcastCols(a, col),
                a -> metal.multiplyBroadcastCols(a, col), a -> metal.multiplyBroadcastRows(a, row),
                a -> metal.causalMask(a, 4), a -> metal.causalSoftmax(a, 4, 0.5f),
                a -> metal.rotary(a, Tensor.ones(2, 2), new Tensor(2, 2), 2, false));
        for (var operation : operations) { independent(input, operation.apply(input)); }
    }

    @Test
    void normalizationAndFusedActivationOutputsHaveIndependentStorageAndGradients() {
        Tensor input = Tensor.ones(2, 4), gain = Tensor.ones(1, 4);
        var rms = metal.rmsNorm(input, gain, 1e-6f);
        independent(input, rms.output());
        independent(input, rms.normalized());
        assertArrayEquals(new int[]{2, 1}, rms.rms().shape());
        independent(input, metal.rmsNormBackward(input, rms.normalized(), rms.rms(), gain));
        independent(input, metal.layerNormBackward(input, input, Tensor.ones(2, 1), 4));
        checkSwiGlu(input, Tensor.ones(2, 4));
        checkSwiGlu(Tensor.ones(DType.FLOAT32, 2, 1, 4), Tensor.ones(DType.FLOAT32, 2, 1, 4));
    }

    @Test
    void groupedAttentionUsesQueryAndCompactKeyValueShapesIndependently() {
        Tensor query = Tensor.ones(512, 64), key = Tensor.ones(256, 64), value = Tensor.ones(256, 64);
        assumeTrue(metal.supportsCausalAttention(query, key, value, 2, 1, 0.125f));
        var saved = metal.causalAttention(query, key, value, 2, 1, 0.125f);
        independent(query, saved.output());
        assertArrayEquals(new int[]{512, 2}, saved.normalization().shape());
        var gradients = metal.causalAttentionBackward(query, key, value, saved, query, 2, 1, 0.125f);
        independent(query, gradients.query());
        independent(key, gradients.key());
        independent(value, gradients.value());
        assertEquals(1, saved.output().get(0, 0), 1e-6f);
    }

    private void checkSwiGlu(Tensor gate, Tensor up) {
        independent(gate, metal.swiGlu(gate, up));
        var gradients = metal.swiGluBackward(gate, gate, up);
        independent(gate, gradients.gateGradient());
        independent(up, gradients.upGradient());
    }

    private static void independent(Tensor source, Tensor output) {
        assertSame(source.layout(), output.layout());
        assertNotSame(source.memory(), output.memory());
        assertNotSame(source.gradientState(), output.gradientState());
        assertFalse(output.memory().hasHostStorage());
        assertNotSame(source.getGpuTag(), output.getGpuTag());
        long version = output.gradientState().version();
        source.gradientState().beforeMutation();
        assertEquals(version, output.gradientState().version());
    }
}
