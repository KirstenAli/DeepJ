package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class GraphBindingIdentityTest extends ComputeGraphTestSupport {
    @Test
    void repeatedStorageAndAliasLookupsReuseTheSameLiveReference() {
        Tensor source = graph.createOutputTensor(graph.newOutputBuffer(2, 3));
        Tensor alias = Tensor.aliasStorage(source);
        GpuBuffer buffer = graph.ensureGpuBuffer(source);
        var reference = graph.bufIdToMemory.get(buffer.id);
        for (int index = 0; index < 1000; index++) {
            assertSame(buffer, graph.ensureStorageBuffer(index % 2 == 0 ? source : alias));
            assertSame(reference, graph.bufIdToMemory.get(buffer.id));
        }
        assertSame(source.memory(), reference.get());
    }

    @Test
    void clearedReferenceIsReplacedBeforeStorageIsReused() {
        Tensor source = graph.createOutputTensor(graph.newOutputBuffer(1, 2));
        GpuBuffer buffer = graph.ensureGpuBuffer(source);
        var reference = graph.bufIdToMemory.get(buffer.id);
        reference.clear();
        assertSame(buffer, graph.ensureGpuBuffer(source));
        assertNotSame(reference, graph.bufIdToMemory.get(buffer.id));
        assertSame(source.memory(), graph.bufIdToMemory.get(buffer.id).get());
    }

    @Test
    void differentStorageCannotReplaceALiveBinding() {
        Tensor source = graph.createOutputTensor(graph.newOutputBuffer(1, 2));
        GpuBuffer buffer = graph.ensureGpuBuffer(source);
        var reference = graph.bufIdToMemory.get(buffer.id);
        assertThrows(IllegalArgumentException.class, () -> graph.bindTensorToBuffer(Tensor.ones(1, 2), buffer));
        assertSame(reference, graph.bufIdToMemory.get(buffer.id));
        assertSame(source.memory(), reference.get());
    }

    @Test
    void stridedAliasesKeepThePhysicalOwnerAndLiveReference() {
        Tensor source = graph.createOutputTensor(graph.newOutputBuffer(2, 3));
        Tensor alias = Tensor.aliasStorage(source, LayoutTransforms.permute(source.layout(), 1, 0));
        GpuBuffer buffer = graph.ensureStorageBuffer(source);
        var reference = graph.bufIdToMemory.get(buffer.id);
        assertSame(buffer, graph.ensureStorageBuffer(alias));
        assertSame(reference, graph.bufIdToMemory.get(buffer.id));
        graph.bindTensorToBuffer(alias, buffer);
        assertSame(reference, graph.bufIdToMemory.get(buffer.id));
    }
}
