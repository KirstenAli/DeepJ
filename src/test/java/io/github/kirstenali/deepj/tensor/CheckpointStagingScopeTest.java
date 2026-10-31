package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.data.*;
import io.github.kirstenali.deepj.optimisers.*;
import io.github.kirstenali.deepj.persistence.TrainingCheckpoint;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.training.*;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.nio.file.*;
import java.util.List;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class CheckpointStagingScopeTest {
    @TempDir Path directory;
    private ComputeGraph graph;

    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @AfterEach void cleanup() {
        if (graph != null) graph.releaseAll();
        Tensor.setBackend(new CpuBackend());
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void restoredStateKeepsOnlyIndependentRetainedDeviceBuffers(DType dtype) throws Exception {
        var source = parameters(dtype, 1);
        var optimizer = trained(source);
        var expected = optimizer.state(source);
        Path file = save(source, optimizer);
        var target = parameters(dtype, 9);
        var restored = AdamW.defaultAdamW(1e-4f);
        device();
        TrainingCheckpoint.load(target, restored, new Cursor(), schedule(), file);
        assertTrue(graph.pendingUploadData.isEmpty());
        assertEquals(dtype == DType.FLOAT32 ? 4 : 6, graph.allocatedBufferIds.size());
        compareState(expected, restored.state(target));
        for (int index = 0; index < source.size(); index++) {
            assertArrayEquals(source.get(index).value.toFloatArray(), target.get(index).value.toFloatArray());
        }
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void malformedFilesRetireProspectiveStateWithoutChangingTargetValues(DType dtype) throws Exception {
        var source = parameters(dtype, 1);
        Path file = save(source, trained(source));
        Files.write(file, new byte[]{1}, StandardOpenOption.APPEND);
        var target = parameters(dtype, 9);
        float[] expected = target.get(0).value.toFloatArray();
        device();
        assertThrows(IOException.class, () -> TrainingCheckpoint.load(target,
                AdamW.defaultAdamW(1e-4f), new Cursor(), schedule(), file));
        assertArrayEquals(expected, target.get(0).value.toFloatArray());
        assertTrue(graph.pendingUploadData.isEmpty());
        assertTrue(graph.allocatedBufferIds.isEmpty());
    }

    private static void compareState(AdamW.State expected, AdamW.State actual) {
        assertEquals(expected.step(), actual.step());
        for (int index = 0; index < expected.firstMoments().size(); index++) {
            compareMoment(expected.firstMoments().get(index), actual.firstMoments().get(index));
            compareMoment(expected.secondMoments().get(index), actual.secondMoments().get(index));
        }
        assertEquals(expected.masterWeights().keySet(), actual.masterWeights().keySet());
        expected.masterWeights().forEach((index, master) -> compareMoment(master, actual.masterWeights().get(index)));
    }

    private static void compareMoment(Tensor expected, Tensor actual) {
        assertTrue(actual.retainsDeviceBuffer());
        assertTrue(((GpuBuffer) actual.getGpuTag()).allocatedOnGpu);
        assertNull(actual.rawData());
        assertArrayEquals(expected.toFloatArray(), actual.toFloatArray());
    }

    private void device() {
        graph = new ComputeGraph(new GpuStorageRuntime());
        var cpu = new CpuBackend();
        Tensor.setBackend((TensorBackend) Proxy.newProxyInstance(TensorBackend.class.getClassLoader(),
                new Class<?>[]{TensorBackend.class}, (proxy, method, arguments) -> invoke(cpu, method, arguments)));
    }

    private Object invoke(CpuBackend cpu, Method method, Object[] arguments) throws Throwable {
        return switch (method.getName()) {
            case "usesDeviceStorage" -> true;
            case "copy" -> graph.copyToDevice((Tensor) arguments[0]);
            case "releaseResources" -> {
                graph.releaseAll();
                yield null;
            }
            default -> invokeCpu(cpu, method, arguments);
        };
    }

    private static Object invokeCpu(CpuBackend cpu, Method method, Object[] arguments) throws Throwable {
        try { return method.invoke(cpu, arguments); }
        catch (InvocationTargetException failure) { throw failure.getCause(); }
    }

    private static List<Parameter> parameters(DType dtype, float first) {
        return List.of(new Parameter(Tensor.from(new float[]{first, 2, 3, 4}, dtype, 2, 2)),
                new Parameter(Tensor.from(new float[]{5, 6, 7, 8}, dtype, 2, 2)));
    }

    private static AdamW trained(List<Parameter> parameters) {
        var optimizer = AdamW.defaultAdamW(1e-4f);
        for (var parameter : parameters) {
            parameter.setGradient(Tensor.from(new float[]{2, -3, 4, -1}, DType.FLOAT32, 2, 2));
        }
        optimizer.step(parameters);
        return optimizer;
    }

    private Path save(List<Parameter> parameters, AdamW optimizer) throws IOException {
        Path file = directory.resolve("training.dj");
        TrainingCheckpoint.save(parameters, optimizer, new Cursor(), new TrainingProgress(1, 2, 2), schedule(), file);
        return file;
    }

    private static CosineLearningRateSchedule schedule() {
        return new CosineLearningRateSchedule(1e-4f, 1e-5f, 1, 10);
    }

    private static final class Cursor implements StatefulBatchSource {
        @Override public Batch nextBatch(int size) { throw new UnsupportedOperationException("Restore only"); }
        @Override public long randomState() { return 0; }
        @Override public void restoreRandomState(long state) { assertEquals(0, state); }
    }
}
