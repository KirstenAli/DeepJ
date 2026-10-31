package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.persistence.ModelSerializer;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class LazyTensorStorageTest extends ComputeGraphTestSupport {
    @TempDir Path directory;
    private TensorBackend previous;

    @BeforeEach void saveBackend() { previous = Tensor.backend(); }
    @AfterEach void restoreBackend() { Tensor.setBackend(previous); }

    @Test
    void publicCpuConstructorsRemainEagerButGpuOutputsDoNotAllocateShadowArrays() {
        assertArrayEquals(new float[4], new Tensor(1, 4).rawData());
        Tensor output = graph.createOutputTensor(graph.newOutputBuffer(1024, 16384));
        assertNull(output.rawData());
        assertTrue(runtime.downloads.isEmpty());
        assertEquals(1024, output.matrixRows());
        assertEquals(16384, output.matrixCols());
    }

    @Test
    void explicitHostReadsDownloadOnceAndPopulateTheRawField() {
        Tensor output = output();
        assertArrayEquals(new float[]{3, 5}, output.data());
        assertSame(output.rawData(), output.data());
        assertArrayEquals(new float[]{3, 5}, output.rowData(0));
        assertEquals(1, runtime.downloads.size());
    }

    @Test
    void cpuFallbackUsesTheOwningGraphEvenAfterTheGlobalBackendChanges() {
        Tensor output = output();
        Tensor.setBackend(new CpuBackend());
        assertArrayEquals(new float[]{4, 6}, output.add(Tensor.ones(1, 2)).data());
        assertEquals(1, runtime.downloads.size());
        output.addScalarInPlace(2);
        assertArrayEquals(new float[]{5, 7}, output.data());
        assertTrue(((GpuBuffer) output.getGpuTag()).needsUpload);
    }

    @Test
    void deviceResultsCanMoveGraphsWithoutOldCleanupInvalidatingTheNewBinding() {
        Tensor output = output();
        var otherRuntime = new RecordingRuntime();
        var other = new ComputeGraph(otherRuntime);
        GpuBuffer moved = other.ensureGpuBuffer(output);
        graph.releaseAll();
        assertSame(moved, output.getGpuTag());
        other.flush();
        assertArrayEquals(new float[]{3, 5}, otherRuntime.uploads.get(0).data());
        assertEquals(1, runtime.downloads.size());
    }

    @Test
    void anotherGraphDelegatesHostReadsToTheBufferOwner() {
        Tensor output = output();
        var otherRuntime = new RecordingRuntime();
        new ComputeGraph(otherRuntime).materialize(output);
        assertArrayEquals(new float[]{3, 5}, output.rawData());
        assertEquals(1, runtime.downloads.size());
        assertTrue(otherRuntime.downloads.isEmpty());
    }

    @Test
    void bindingAnExistingDeviceBufferToAnotherGraphIsRejected() {
        Tensor output = output();
        var other = new ComputeGraph(new RecordingRuntime());
        assertThrows(IllegalArgumentException.class,
                () -> other.bindTensorToBuffer(output, (GpuBuffer) output.getGpuTag()));
    }

    @Test
    void releaseAllMaterializesLazyValuesBeforeFreeingTheirGpuStorage() {
        Tensor output = output();
        graph.releaseAll();
        assertNull(output.getGpuTag());
        assertArrayEquals(new float[]{3, 5}, output.rawData());
        assertEquals(1, runtime.downloads.size());
    }

    @Test
    void uploadSnapshotsOfImplicitZerosDoNotCreatePersistentHostStorage() {
        Tensor tensor = Tensor.deviceOnly(1, 4);
        float[] first = TensorAdapters.packF32(tensor);
        first[0] = 9;
        assertArrayEquals(new float[4], TensorAdapters.packF32(tensor));
        assertNull(tensor.rawData());
        assertArrayEquals(new float[4], tensor.data());
    }

    @Test
    void copyingAndSerializationMaterializeGpuResults() throws IOException {
        Tensor output = output();
        assertArrayEquals(new float[]{3, 5}, new Tensor(output).rawData());
        Path file = directory.resolve("weights.dj");
        ModelSerializer.save(List.of(new Parameter(output)), file);
        Parameter target = new Parameter(Tensor.deviceOnly(1, 2));
        ModelSerializer.load(List.of(target), file);
        assertArrayEquals(new float[]{3, 5}, target.value.data());
    }

    @Test
    void managedWritesAndUnpackingWorkWithoutEagerHostStorage() {
        Tensor output = output();
        output.set(0, 1, 8);
        assertArrayEquals(new float[]{3, 8}, output.data());
        Tensor second = output();
        TensorAdapters.unpackF32Into(new float[]{11, 13}, second);
        assertArrayEquals(new float[]{11, 13}, second.data());
        assertTrue(((GpuBuffer) second.getGpuTag()).needsUpload);
    }

    private Tensor output() {
        runtime.downloadResult = new float[]{3, 5};
        return graph.createOutputTensor(graph.newOutputBuffer(1, 2));
    }
}
