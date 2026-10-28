package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.lang.reflect.Proxy;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class HostStagingTest {
    @AfterEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest
    @EnumSource(DType.class)
    void privatePayloadIsHandedOffWithoutASecondArrayWhilePublicCopiesRemainIndependent(DType dtype) {
        var graph = new ComputeGraph(new GpuStorageRuntime());
        Tensor source = Tensor.from(new float[]{-0.0f, 2, 3, 4}, dtype, 2, 2);
        Object original = dtype == DType.FLOAT32 ? source.memory().floats() : source.memory().shorts();
        copyingBackend(graph);
        Tensor copy = HostStaging.copy(source);
        assertSame(original, graph.pendingUploadData.get(0));
        assertFalse(HostStaging.owns(source));
        source.memory().discardStaleHost();
        graph.releaseAll();
        assertArrayEquals(new float[]{-0.0f, 2, 3, 4}, copy.toFloatArray());
        Tensor ordinary = graph.copyToDevice(source);
        assertNotSame(original, graph.pendingUploadData.get(0));
        graph.releaseAll();
        assertArrayEquals(new float[]{-0.0f, 2, 3, 4}, ordinary.toFloatArray());
    }

    @Test
    void ownershipIsRestoredAfterNestedAndFailedCopies() {
        Tensor outer = new Tensor(1), inner = new Tensor(1);
        Tensor.setBackend((TensorBackend) Proxy.newProxyInstance(TensorBackend.class.getClassLoader(),
                new Class<?>[]{TensorBackend.class}, (proxy, method, arguments) -> nested(outer, inner, arguments)));
        HostStaging.copy(outer);
        assertFalse(HostStaging.owns(outer));
        assertFalse(HostStaging.owns(inner));
    }

    private static Tensor nested(Tensor outer, Tensor inner, Object[] arguments) {
        Tensor source = (Tensor) arguments[0];
        assertTrue(HostStaging.owns(source));
        if (source == inner) throw new IllegalStateException("Controlled copy failure");
        assertThrows(IllegalStateException.class, () -> HostStaging.copy(inner));
        assertTrue(HostStaging.owns(outer));
        return outer;
    }

    @Test
    void partialViewsAreCopiedRatherThanHandingOffTheirEntireBackingArray() {
        var graph = new ComputeGraph(new GpuStorageRuntime());
        Tensor source = Tensor.from(new float[]{1, 2, 3, 4}, DType.FLOAT32, 2, 2);
        Tensor partial = Tensor.aliasStorage(source, LayoutTransforms.select(source.layout(), 0, 1));
        copyingBackend(graph);
        Tensor copy = HostStaging.copy(partial);
        assertNotSame(source.rawData(), graph.pendingUploadData.get(0));
        assertEquals(2, ((float[]) graph.pendingUploadData.get(0)).length);
        graph.releaseAll();
        assertArrayEquals(new float[]{3, 4}, copy.toFloatArray());
    }

    private static void copyingBackend(ComputeGraph graph) {
        Tensor.setBackend((TensorBackend) Proxy.newProxyInstance(TensorBackend.class.getClassLoader(),
                new Class<?>[]{TensorBackend.class}, (proxy, method, arguments) -> {
                    if (!method.getName().equals("copy")) throw new UnsupportedOperationException(method.getName());
                    return graph.copyToDevice((Tensor) arguments[0]);
                }));
    }
}
