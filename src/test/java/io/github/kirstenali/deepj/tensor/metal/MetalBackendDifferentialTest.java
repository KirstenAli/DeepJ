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

class MetalBackendDifferentialTest extends MetalDifferentialTestSupport {

    @Test
    void elementwiseBinaryOperationsMatchCpu() {
        Tensor a = random(5, 7, 1L);
        Tensor b = positive(5, 7, 2L);
        compareBinary(a, b, TensorBackend::add, 1e-5f, 1e-5f);
        compareBinary(a, b, TensorBackend::subtract, 1e-5f, 1e-5f);
        compareBinary(a, b, TensorBackend::multiply, 1e-5f, 1e-5f);
        compareBinary(a, b, TensorBackend::divide, 1e-5f, 1e-5f);
    }

    @Test
    void unaryOperationsMatchCpu() {
        Tensor signed = random(5, 7, 3L);
        Tensor positive = positive(5, 7, 4L);
        compareUnary(signed, TensorBackend::neg, 1e-5f, 1e-5f);
        compareUnary(signed, TensorBackend::exp, 1e-4f, 1e-4f);
        compareUnary(positive, TensorBackend::log, 1e-4f, 1e-4f);
        compareUnary(positive, TensorBackend::sqrt, 1e-4f, 1e-4f);
        compareUnary(signed, (backend, value) -> backend.pow(value, 2.0f), 1e-4f, 1e-4f);
        compareUnary(signed, (backend, value) -> backend.clamp(value, -0.1f, 0.2f), 1e-5f, 1e-5f);
    }

    @Test
    void activationsAndTheirBackwardPassesMatchCpu() {
        Tensor input = random(5, 7, 5L);
        Tensor upstream = random(5, 7, 6L);
        compareUnary(input, TensorBackend::relu, 1e-5f, 1e-5f);
        compareUnary(input, TensorBackend::gelu, 1e-4f, 1e-4f);
        compareUnary(input, TensorBackend::tanh, 1e-4f, 1e-4f);
        compareUnary(input, TensorBackend::sigmoid, 1e-4f, 1e-4f);
        compareBinary(input, upstream, TensorBackend::reluBackward, 1e-5f, 1e-5f);
        compareBinary(input, upstream, TensorBackend::geluBackward, 2e-4f, 2e-4f);
    }

    @Test
    void inPlaceOperationsMatchCpu() {
        Tensor input = random(5, 7, 7L);
        Tensor positive = positive(5, 7, 8L);
        compareInPlaceBinary(input, positive, TensorBackend::addInPlace);
        compareInPlaceBinary(input, positive, TensorBackend::subtractInPlace);
        compareInPlaceBinary(input, positive, TensorBackend::multiplyInPlace);
        compareInPlaceBinary(input, positive, TensorBackend::divideInPlace);
        compareInPlaceUnary(input, TensorBackend::geluInPlace);
        compareInPlaceUnary(input, TensorBackend::sigmoidInPlace);
    }

    @Test
    void queuedCrossEntropyTargetsRemainIndependent() {
        Tensor logits = random(3, 5, 9L);
        Tensor expectedA = cpu.crossEntropyGradient(logits, new int[]{0, 1, 2});
        Tensor expectedB = cpu.crossEntropyGradient(logits, new int[]{4, 3, 2});
        Tensor actualA = metal.crossEntropyGradient(new Tensor(logits), new int[]{0, 1, 2});
        Tensor actualB = metal.crossEntropyGradient(new Tensor(logits), new int[]{4, 3, 2});
        assertTensorClose(expectedA, actualA, 1e-3f, 1e-2f);
        assertTensorClose(expectedB, actualB, 1e-3f, 1e-2f);
    }

    @Test
    void maskedCrossEntropyMatchesCpu() {
        Tensor logits = random(5, 7, 10L);
        int[] targets = {0, 1, 2, 3, 4};
        boolean[] mask = {false, true, false, true, true};
        float expectedLoss = cpu.crossEntropyLoss(new Tensor(logits), targets, mask);
        float actualLoss = metal.crossEntropyLoss(new Tensor(logits), targets, mask);
        Tensor expectedGradient = cpu.crossEntropyGradient(new Tensor(logits), targets, mask);
        Tensor actualGradient = metal.crossEntropyGradient(new Tensor(logits), targets, mask);
        assertEquals(expectedLoss, actualLoss, 1e-3f);
        assertTensorClose(expectedGradient, actualGradient, 1e-3f, 1e-2f);
    }

    @Test
    void fusedCrossEntropyMatchesCpu() {
        Tensor logits = random(17, 769, 15L);
        int[] targets = randomTargets(logits.rows, logits.cols, 16L);
        float expectedLoss = cpu.crossEntropyLoss(new Tensor(logits), targets);
        Tensor expectedGradient = cpu.crossEntropyGradient(new Tensor(logits), targets);
        CrossEntropyResult actual = metal.crossEntropy(new Tensor(logits), targets);
        assertEquals(expectedLoss, actual.meanLoss(), 1e-3f);
        assertTensorClose(expectedGradient, actual.gradient(), 1e-3f, 1e-2f);
    }

    @Test
    void fusedRmsNormMatchesCpu() {
        Tensor input = random(11, 37, 17L);
        Tensor gamma = positive(1, input.cols, 18L);
        Tensor gradient = random(input.rows, input.cols, 19L);
        RmsNormResult expected = cpu.rmsNorm(input, gamma, 1e-6f);
        RmsNormResult actual = metal.rmsNorm(new Tensor(input), new Tensor(gamma), 1e-6f);
        assertRmsNormClose(expected, actual);
        Tensor expectedBackward = cpu.rmsNormBackward(gradient,
                expected.normalized(), expected.rms(), gamma);
        Tensor actualBackward = metal.rmsNormBackward(new Tensor(gradient),
                actual.normalized(), actual.rms(), new Tensor(gamma));
        assertTensorClose(expectedBackward, actualBackward, 2e-4f, 2e-4f);
    }

    private static void assertRmsNormClose(RmsNormResult expected, RmsNormResult actual) {
        assertTensorClose(expected.output(), actual.output(), 2e-4f, 2e-4f);
        assertTensorClose(expected.normalized(), actual.normalized(), 2e-4f, 2e-4f);
        assertTensorClose(expected.rms(), actual.rms(), 2e-4f, 2e-4f);
    }

    @Test
    void fusedSwiGluMatchesCpu() {
        Tensor gate = random(19, 31, 33L);
        Tensor up = random(19, 31, 34L);
        Tensor gradient = random(19, 31, 35L);
        assertTensorClose(cpu.swiGlu(gate, up), metal.swiGlu(gate, up), 1e-5f, 1e-5f);
        SwiGluBackwardResult expected = cpu.swiGluBackward(gradient, gate, up);
        SwiGluBackwardResult actual = metal.swiGluBackward(gradient, gate, up);
        assertTensorClose(expected.gateGradient(), actual.gateGradient(), 2e-5f, 2e-5f);
        assertTensorClose(expected.upGradient(), actual.upGradient(), 2e-5f, 2e-5f);
    }

    @Test
    void queuedScatterIndicesRemainIndependent() {
        Tensor expectedA = Tensor.zeros(6, 2);
        Tensor expectedB = Tensor.zeros(6, 2);
        Tensor actualA = Tensor.zeros(6, 2);
        Tensor actualB = Tensor.zeros(6, 2);
        Tensor gradient = random(3, 2, 10L);
        cpu.scatterAddRows(expectedA, new int[]{1, 3, 5}, gradient);
        cpu.scatterAddRows(expectedB, new int[]{0, 2, 4}, gradient);
        metal.scatterAddRows(actualA, new int[]{1, 3, 5}, new Tensor(gradient));
        metal.scatterAddRows(actualB, new int[]{0, 2, 4}, new Tensor(gradient));
        assertTensorClose(expectedA, actualA, 1e-5f, 1e-5f);
        assertTensorClose(expectedB, actualB, 1e-5f, 1e-5f);
    }

    @Test
    void headLayoutOperationsMatchCpu() {
        Tensor input = random(5, 12, 11L);
        Tensor expectedSplit = cpu.splitHeads(new Tensor(input), 3);
        Tensor actualSplit = metal.splitHeads(new Tensor(input), 3);
        assertTensorClose(expectedSplit, actualSplit, 1e-5f, 1e-5f);
        assertTensorClose(cpu.mergeHeads(expectedSplit, 3),
                metal.mergeHeads(actualSplit, 3), 1e-5f, 1e-5f);
    }

    @Test
    void batchedMatmulVariantsMatchCpu() {
        compareBatched(random(12, 5, 12L), random(15, 6, 13L), false, false);
        compareBatched(random(12, 5, 14L), random(18, 5, 15L), false, true);
        compareBatched(random(15, 4, 16L), random(15, 6, 17L), true, false);
        compareBatched(random(15, 4, 18L), random(18, 5, 19L), true, true);
    }

    @Test
    void causalAttentionRotaryAndGatherMatchCpu() {
        compareCausalMask();
        compareCausalSoftmax();
        compareRotary(false);
        compareRotary(true);
        compareGather();
    }

    private void compareBatched(Tensor left, Tensor right,
                                boolean transposeLeft, boolean transposeRight) {
        Tensor expected = cpu.batchedMatmul(new Tensor(left), new Tensor(right),
                3, transposeLeft, transposeRight);
        Tensor actual = metal.batchedMatmul(new Tensor(left), new Tensor(right),
                3, transposeLeft, transposeRight);
        assertTensorClose(expected, actual, 1e-4f, 1e-4f);
    }

    private void compareCausalMask() {
        Tensor input = random(12, 4, 20L);
        assertTensorClose(cpu.causalMask(new Tensor(input), 4),
                metal.causalMask(new Tensor(input), 4), 1e-5f, 1e-5f);
    }

    private void compareCausalSoftmax() {
        Tensor input = random(12, 4, 28L);
        assertTensorClose(cpu.causalSoftmax(new Tensor(input), 4, 0.25f),
                metal.causalSoftmax(new Tensor(input), 4, 0.25f), 1e-5f, 1e-5f);
    }

    private void compareRotary(boolean inverse) {
        Tensor input = random(12, 6, inverse ? 25L : 21L);
        Tensor cosine = random(4, 3, inverse ? 26L : 22L);
        Tensor sine = random(4, 3, inverse ? 27L : 23L);
        Tensor expected = cpu.rotary(input, cosine, sine, 4, inverse);
        Tensor actual = metal.rotary(new Tensor(input), new Tensor(cosine),
                new Tensor(sine), 4, inverse);
        assertTensorClose(expected, actual, 1e-5f, 1e-5f);
    }

    private void compareGather() {
        Tensor input = random(8, 5, 24L);
        int[] rows = {7, 1, 4, 1};
        assertTensorClose(cpu.sliceRows(input, rows),
                metal.sliceRows(new Tensor(input), rows), 1e-5f, 1e-5f);
    }

    private void compareBinary(Tensor a, Tensor b, BinaryOperation operation,
                               float absoluteTolerance, float relativeTolerance) {
        Tensor expected = operation.apply(cpu, new Tensor(a), new Tensor(b));
        Tensor actual = operation.apply(metal, new Tensor(a), new Tensor(b));
        assertTensorClose(expected, actual, absoluteTolerance, relativeTolerance);
    }

    private void compareUnary(Tensor input, UnaryOperation operation,
                              float absoluteTolerance, float relativeTolerance) {
        Tensor expected = operation.apply(cpu, new Tensor(input));
        Tensor actual = operation.apply(metal, new Tensor(input));
        assertTensorClose(expected, actual, absoluteTolerance, relativeTolerance);
    }

    private void compareInPlaceBinary(Tensor a, Tensor b, BinaryInPlaceOperation operation) {
        Tensor expected = new Tensor(a);
        Tensor actual = new Tensor(a);
        operation.apply(cpu, expected, new Tensor(b));
        operation.apply(metal, actual, new Tensor(b));
        assertTensorClose(expected, actual, 1e-4f, 1e-4f);
    }

    private void compareInPlaceUnary(Tensor input, UnaryInPlaceOperation operation) {
        Tensor expected = new Tensor(input);
        Tensor actual = new Tensor(input);
        operation.apply(cpu, expected);
        operation.apply(metal, actual);
        assertTensorClose(expected, actual, 1e-4f, 1e-4f);
    }


}
