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

abstract class MetalDifferentialTestSupport {

    protected CpuBackend cpu;
    protected MetalBackend metal;
    protected TensorBackend previous;

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

    protected Tensor random(int rows, int cols, long seed) {
        return cpu.random(rows, cols, new Random(seed));
    }

    protected Tensor positive(int rows, int cols, long seed) {
        return cpu.addScalar(random(rows, cols, seed), 1.0f);
    }

    protected static int[] randomTargets(int rows, int columns, long seed) {
        Random random = new Random(seed);
        int[] targets = new int[rows];
        for (int row = 0; row < rows; row++) {
            targets[row] = random.nextInt(columns);
        }
        return targets;
    }

    protected static List<Tensor> copyGradients(List<Parameter> parameters) {
        List<Tensor> gradients = new ArrayList<>(parameters.size());
        for (Parameter parameter : parameters) {
            gradients.add(new Tensor(parameter.grad));
        }
        return gradients;
    }

    protected static void assertGradientListsClose(List<Tensor> expected, List<Tensor> actual,
                                                 float absoluteTolerance, float relativeTolerance) {
        assertEquals(expected.size(), actual.size(), "parameter count");
        for (int i = 0; i < expected.size(); i++) {
            assertTensorClose(expected.get(i), actual.get(i), absoluteTolerance, relativeTolerance,
                    "parameter " + i);
        }
    }

    protected static void assertTensorClose(Tensor expected, Tensor actual,
                                          float absoluteTolerance, float relativeTolerance) {
        assertTensorClose(expected, actual, absoluteTolerance, relativeTolerance, "tensor");
    }

    protected static void assertTensorClose(Tensor expected, Tensor actual, float absoluteTolerance,
                                          float relativeTolerance, String label) {
        expected.materialize();
        actual.materialize();
        assertEquals(expected.rows, actual.rows, "row count");
        assertEquals(expected.cols, actual.cols, "column count");
        for (int i = 0; i < expected.data.length; i++) {
            assertElementClose(expected.data[i], actual.data[i], absoluteTolerance, relativeTolerance, label, i);
        }
    }

    protected static void assertElementClose(float expected, float actual, float absoluteTolerance,
                                           float relativeTolerance, String label, int index) {
        float tolerance = absoluteTolerance + relativeTolerance * Math.abs(expected);
        assertTrue(Float.isFinite(actual), label + " has non-finite Metal value at flat index " + index);
        assertTrue(Math.abs(expected - actual) <= tolerance,
                label + " flat index " + index + " expected=" + expected + " actual=" + actual);
    }

    @FunctionalInterface
    protected interface BinaryOperation {
        Tensor apply(TensorBackend backend, Tensor a, Tensor b);
    }

    @FunctionalInterface
    protected interface UnaryOperation {
        Tensor apply(TensorBackend backend, Tensor value);
    }

    @FunctionalInterface
    protected interface BinaryInPlaceOperation {
        void apply(TensorBackend backend, Tensor a, Tensor b);
    }

    @FunctionalInterface
    protected interface UnaryInPlaceOperation {
        void apply(TensorBackend backend, Tensor value);
    }

    protected record ModelResult(Tensor output, List<Tensor> gradients) {}
}
