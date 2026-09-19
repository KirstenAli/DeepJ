package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.data.BatchSource;
import io.github.kirstenali.deepj.models.DecoderOnlyModel;
import io.github.kirstenali.deepj.models.deepseek.DeepSeekConfig;
import io.github.kirstenali.deepj.models.deepseek.DeepSeekModel;
import io.github.kirstenali.deepj.models.gpt.GPTConfig;
import io.github.kirstenali.deepj.models.gpt.GPTModel;
import io.github.kirstenali.deepj.models.llama.LlamaConfig;
import io.github.kirstenali.deepj.models.llama.LlamaModel;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.CrossEntropyResult;
import io.github.kirstenali.deepj.tensor.RmsNormResult;
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

class MetalBackendDifferentialTest {

    private CpuBackend cpu;
    private MetalBackend metal;
    private TensorBackend previous;

    @BeforeEach
    void setUp() {
        assumeTrue(MetalBackend.isAvailable(), "Metal device not available");
        previous = Tensor.backend();
        cpu = new CpuBackend();
        metal = new MetalBackend();
        Tensor.setBackend(metal);
    }

    @AfterEach
    void tearDown() {
        if (metal != null) metal.releaseResources();
        if (previous != null) Tensor.setBackend(previous);
    }

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

    @Test
    void gptForwardBackwardAndParameterGradientsMatchCpu() {
        compareModel(() -> new GPTModel(new GPTConfig(11, 4, 4, 2, 1, 6), 21L));
    }

    @Test
    void llamaForwardBackwardAndParameterGradientsMatchCpu() {
        compareModel(() -> new LlamaModel(new LlamaConfig(11, 4, 4, 2, 1, 8), 22L));
    }

    @Test
    void deepSeekForwardBackwardAndParameterGradientsMatchCpu() {
        compareModel(() -> new DeepSeekModel(new DeepSeekConfig(11, 4, 4, 2, 1, 8, 3, 2), 23L));
    }

    @Test
    void deepSeekForwardBackwardDoesNotDownloadIntermediateTensors() {
        AtomicInteger downloads = new AtomicInteger();
        var config = new DeepSeekConfig(11, 4, 4, 2, 1, 8, 3, 2);
        DecoderOnlyModel model = new DeepSeekModel(config, 25L);
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
    void temporaryReleaseDoesNotChangeRepeatedDeepSeekTraining() {
        var config = new DeepSeekConfig(11, 4, 4, 2, 1, 8, 3, 2);
        MetalBackend baseline = new MetalBackend();
        Tensor.setBackend(baseline);
        DecoderOnlyModel expected = new DeepSeekModel(config, 24L);
        Tensor.setBackend(metal);
        DecoderOnlyModel actual = new DeepSeekModel(config, 24L);
        float expectedLoss = trainRepeated(baseline, expected, 0);
        float actualLoss = trainRepeated(metal, actual, 1);
        assertEquals(expectedLoss, actualLoss, 1e-5f);
        assertParameterValuesClose(expected.parameters(), actual.parameters(), 1e-5f);
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

    private Tensor random(int rows, int cols, long seed) {
        return cpu.random(rows, cols, new Random(seed));
    }

    private Tensor positive(int rows, int cols, long seed) {
        return cpu.addScalar(random(rows, cols, seed), 1.0f);
    }

    private static int[] randomTargets(int rows, int columns, long seed) {
        Random random = new Random(seed);
        int[] targets = new int[rows];
        for (int row = 0; row < rows; row++) targets[row] = random.nextInt(columns);
        return targets;
    }

    private static List<Tensor> copyGradients(List<Parameter> parameters) {
        List<Tensor> gradients = new ArrayList<>(parameters.size());
        for (Parameter parameter : parameters) gradients.add(new Tensor(parameter.grad));
        return gradients;
    }

    private static void assertGradientListsClose(List<Tensor> expected, List<Tensor> actual,
                                                 float absoluteTolerance, float relativeTolerance) {
        assertEquals(expected.size(), actual.size(), "parameter count");
        for (int i = 0; i < expected.size(); i++) {
            assertTensorClose(expected.get(i), actual.get(i), absoluteTolerance, relativeTolerance,
                    "parameter " + i);
        }
    }

    private static void assertTensorClose(Tensor expected, Tensor actual,
                                          float absoluteTolerance, float relativeTolerance) {
        assertTensorClose(expected, actual, absoluteTolerance, relativeTolerance, "tensor");
    }

    private static void assertTensorClose(Tensor expected, Tensor actual, float absoluteTolerance,
                                          float relativeTolerance, String label) {
        expected.materialize();
        actual.materialize();
        assertEquals(expected.rows, actual.rows, "row count");
        assertEquals(expected.cols, actual.cols, "column count");
        for (int i = 0; i < expected.data.length; i++) {
            assertElementClose(expected.data[i], actual.data[i], absoluteTolerance, relativeTolerance, label, i);
        }
    }

    private static void assertElementClose(float expected, float actual, float absoluteTolerance,
                                           float relativeTolerance, String label, int index) {
        float tolerance = absoluteTolerance + relativeTolerance * Math.abs(expected);
        assertTrue(Float.isFinite(actual), label + " has non-finite Metal value at flat index " + index);
        assertTrue(Math.abs(expected - actual) <= tolerance,
                label + " flat index " + index + " expected=" + expected + " actual=" + actual);
    }

    @FunctionalInterface
    private interface BinaryOperation {
        Tensor apply(TensorBackend backend, Tensor a, Tensor b);
    }

    @FunctionalInterface
    private interface UnaryOperation {
        Tensor apply(TensorBackend backend, Tensor value);
    }

    @FunctionalInterface
    private interface BinaryInPlaceOperation {
        void apply(TensorBackend backend, Tensor a, Tensor b);
    }

    @FunctionalInterface
    private interface UnaryInPlaceOperation {
        void apply(TensorBackend backend, Tensor value);
    }

    private record ModelResult(Tensor output, List<Tensor> gradients) {}
}
