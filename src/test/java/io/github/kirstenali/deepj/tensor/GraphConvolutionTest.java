package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class GraphConvolutionTest extends ComputeGraphTestSupport {
    @Test void invalidShapesAreRejectedBeforeRecordingCommands() {
        assertThrows(IllegalArgumentException.class,
                () -> graph.convolution2D(new Tensor(1, 2, 3, 5), new Tensor(4, 2, 2, 3), 1));
        assertEquals(0, graph.cmdPos);
    }

    @Test void commandsRetainTypedStorageArgumentsAndTheCompleteConvolutionGeometry() {
        graph = new ComputeGraph(new RecordingRuntime() {
            @Override public boolean supportsLayouts() { return true; }
        });
        Tensor input = new Tensor(2, 4, 5, 7), weight = new Tensor(6, 2, 3, 5);
        Tensor output = graph.convolution2D(input, weight, 2);
        assertArrayEquals(new int[]{2, 6, 5, 7}, output.shape());
        int[] command = java.util.Arrays.copyOf(graph.cmdStream, graph.cmdPos);
        assertEquals(GraphOpMetadata.OP_CONVOLUTION_2D, command[0]);
        assertArrayEquals(new int[]{2, 4, 5, 7, 6, 3, 5, 2, 1},
                java.util.Arrays.copyOfRange(command, 4, command.length));
        var metadata = GraphOpMetadata.OP_METADATA[command[0]];
        assertEquals(13, metadata.stride());
        assertArrayEquals(new int[]{1, 2, 3}, metadata.bufferArgOffsets());
    }
}
