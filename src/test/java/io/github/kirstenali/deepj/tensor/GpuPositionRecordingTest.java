package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;
import java.util.Arrays;
import static org.junit.jupiter.api.Assertions.*;

class GpuPositionRecordingTest {
    @Test
    void causalRecordCarriesLogicalShapeScaleModeAndBothLifetimeBufferBindings() {
        ComputeGraph graph = new ComputeGraph(new GpuStorageRuntime());
        Tensor input = new Tensor(DType.FLOAT16, 4, 4);
        Tensor result = graph.layoutCausal(input, 4, 0.625f, 2);
        assertArrayEquals(new int[]{78, id(input), id(result), 0, 4, Float.floatToRawIntBits(0.625f), 2},
                Arrays.copyOf(graph.cmdStream, graph.cmdPos));
        assertArrayEquals(new long[]{2, 16, 16, 0, 4, 4, 4, 1}, graph.layouts.snapshot());
        assertArrayEquals(new int[]{1, 2}, ComputeGraph.OP_METADATA[78].bufferArgOffsets());
        assertEquals(7, ComputeGraph.getOpStride(78));
        assertEquals(DType.FLOAT16, result.dtype());
        assertFalse(TensorTestStorage.hasHostStorage(result));
    }

    @Test
    void rotaryRecordSeparatesAllLayoutsDtypesAndInverseFlagWithoutCopyingInputValues() {
        ComputeGraph graph = new ComputeGraph(new GpuStorageRuntime());
        Tensor input = new Tensor(DType.FLOAT16, 4, 4), cosine = new Tensor(4, 2);
        Tensor sine = new Tensor(DType.BFLOAT16, 4, 2);
        Tensor result = graph.layoutRotary(input, cosine, sine, 4, true);
        assertArrayEquals(new int[]{79, id(input), id(cosine), id(sine), id(result), 0, 8, 16, 4, 1},
                Arrays.copyOf(graph.cmdStream, graph.cmdPos));
        assertEquals(24, graph.layouts.snapshot().length);
        assertArrayEquals(new int[]{1, 2, 3, 4}, ComputeGraph.OP_METADATA[79].bufferArgOffsets());
        assertEquals(10, ComputeGraph.getOpStride(79));
        assertEquals(DType.FLOAT16, result.dtype());
        assertFalse(TensorTestStorage.hasHostStorage(result));
    }

    private static int id(Tensor input) { return ((GpuBuffer) input.getGpuTag()).id; }
}
