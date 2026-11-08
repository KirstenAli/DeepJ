package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.ComputeGraph;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MetalConvolutionValidationTest extends MetalDifferentialTestSupport {
    private int[] command() {
        int input = metal.graph.ensureStorageBuffer(new Tensor(1, 2, 3, 5)).id;
        int weight = metal.graph.ensureStorageBuffer(new Tensor(4, 2, 3, 3)).id;
        int output = metal.graph.ensureStorageBuffer(new Tensor(1, 4, 3, 5)).id;
        metal.graph.flush();
        return new int[]{ComputeGraph.OP_CONVOLUTION_2D, input, weight, output, 1, 2, 3, 5, 4, 3, 3, 1, 1};
    }

    @Test void nativeGeometryValidationRejectsZeroNegativeOverflowAndNonDivisibleDimensions() {
        for (int axis = 4; axis < 13; axis++) {
            invalid(axis, 0);
            invalid(axis, -1);
        }
        invalid(4, Integer.MAX_VALUE);
        invalid(5, 3);
        invalid(9, 2);
        invalid(10, 2);
        invalid(11, 3);
        invalid(12, 2);
        invalid(12, 3);
        invalid(12, 4);
    }

    @Test void nativeValidationRejectsAliasedOutputsAndMissingOrTruncatedBuffers() {
        int[] command = command();
        command[3] = command[1];
        reject(command, command.length);
        command = command();
        command[1] = Integer.MAX_VALUE;
        reject(command, command.length);
        command = command();
        reject(command, command.length - 1);
    }

    private void invalid(int position, int value) {
        int[] command = command();
        command[position] = value;
        reject(command, command.length);
    }

    private static void reject(int[] command, int count) {
        assertThrows(RuntimeException.class, () -> MetalNative.nativeFlushOps(command, count));
    }
}
