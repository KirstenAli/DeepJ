package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.ComputeGraph;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class MetalCompactStreamValidationTest extends MetalCommandValidationSupport {
    @Override protected long[] words() {
        return new long[]{2, 16384, 16384, 0, 64, 256, 256, 1,
                2, 65536, 65536, 0, 256, 256, 256, 1};
    }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void failedStreamsDrainSubmittedPrefixesBeforeBuffersCanBeReused(DType dtype) {
        invalid(failedStream(dtype));
        Tensor result = Tensor.ones(dtype, 64, 256).matmul(Tensor.ones(dtype, 256, 256));
        for (float value : result.toFloatArray()) { assertEquals(256, value); }
    }

    private int[] failedStream(DType dtype) {
        int left = buffer(dtype, 64 * 256), weight = buffer(dtype, 256 * 256);
        int[] stream = new int[20 * 6 + 1];
        for (int step = 0; step < 20; step++) {
            int output = buffer(dtype, 64 * 256);
            int[] command = {ComputeGraph.OP_LAYOUT_MATMUL, left, weight, output, 0, 8};
            System.arraycopy(command, 0, stream, step * 6, command.length);
            left = output;
        }
        stream[stream.length - 1] = Integer.MAX_VALUE;
        return stream;
    }
}
