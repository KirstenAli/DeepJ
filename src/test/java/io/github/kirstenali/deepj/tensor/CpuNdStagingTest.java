package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.testing.NdMatmulReference;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class CpuNdStagingTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest
    @EnumSource(DType.class)
    void transposedBroadcastOperandsMatchIndependentReferenceWithoutMutation(DType dtype) {
        Tensor left = NdMatmulReference.broadcastLeft(dtype).permute(0, 1, 3, 2);
        Tensor right = NdMatmulReference.broadcastRight(dtype).permute(0, 1, 3, 2);
        float[] beforeLeft = left.toFloatArray(), beforeRight = right.toFloatArray();
        NdMatmulReference.assertProduct(left, right, 2, 3, 3, 3);
        assertArrayEquals(beforeLeft, left.toFloatArray());
        assertArrayEquals(beforeRight, right.toFloatArray());
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void serialAndParallelStagingProduceIdenticalResults(DType dtype) {
        Tensor left = NdMatmulReference.broadcastLeft(dtype);
        Tensor right = NdMatmulReference.broadcastRight(dtype);
        boolean parallel = DeepJExecutor.isParallelEnabled();
        try {
            DeepJExecutor.setParallelEnabled(false);
            float[] serial = left.matmul(right).toFloatArray();
            DeepJExecutor.setParallelEnabled(true);
            assertArrayEquals(serial, left.matmul(right).toFloatArray());
        } finally {
            DeepJExecutor.setParallelEnabled(parallel);
        }
    }
}
