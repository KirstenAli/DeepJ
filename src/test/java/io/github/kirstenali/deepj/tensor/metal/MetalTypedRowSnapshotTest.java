package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class MetalTypedRowSnapshotTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(DType.class)
    void requestedHostRowsRespectGpuSharedStorageOffsetsAndStrides(DType dtype) {
        Tensor source = metal.copy(Tensor.from(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, dtype, 4, 3));
        Tensor view = source.slice(0, 1, 4, 2).transpose(0, 1);
        assertFalse(TensorTestStorage.hasHostStorage(source));
        long version = source.gradientState().version();
        float[] row = view.rowData(1);
        assertArrayEquals(new float[]{5, 11}, row);
        assertTrue(TensorTestStorage.hasHostStorage(source));
        assertEquals(version, source.gradientState().version());
        metal.releaseTemporaryResources();
        assertArrayEquals(new float[]{5, 11}, view.rowData(1));
        row[0] = 99;
        assertEquals(5, source.get(1, 1));
    }
}
