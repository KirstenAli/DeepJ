package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.Locale;

/** Manual: per-dtype GEMM timing at training shapes. Run with -Dtest=MetalMatmulDtypeTimingTest. */
@Disabled("manual timing")
class MetalMatmulDtypeTimingTest extends MetalDifferentialTestSupport {
    private static final int ROWS = 1024, WARMUP = 3, ITERATIONS = 10;

    @Test
    void gemmTimingPerOperandDtype() {
        for (int[] shape : new int[][]{{768, 3072}, {3072, 768}, {768, 768}, {768, 16384}}) {
            for (DType left : DType.values()) {
                for (DType right : DType.values()) {
                    time(shape[0], shape[1], left, right);
                }
            }
        }
    }

    private void time(int inner, int columns, DType leftType, DType rightType) {
        Tensor left = operand(leftType, ROWS, inner, 1), right = operand(rightType, inner, columns, 2);
        run(left, right, WARMUP);
        long start = System.nanoTime();
        run(left, right, ITERATIONS);
        double millis = (System.nanoTime() - start) / 1e6 / ITERATIONS;
        System.out.printf(Locale.ROOT, "%5dx%5dx%5d  %-9s x %-9s  %8.3f ms%n",
                ROWS, inner, columns, leftType, rightType, millis);
    }

    /** Chain products through one running sum so a single download covers every kernel. */
    private void run(Tensor left, Tensor right, int count) {
        Tensor sum = metal.cast(metal.matmul(left, right), DType.FLOAT32);
        for (int i = 1; i < count; i++) {
            sum = metal.add(sum, metal.cast(metal.matmul(left, right), DType.FLOAT32));
        }
        metal.sum(sum);
    }

    private Tensor operand(DType dtype, int rows, int columns, long seed) {
        float[] values = random(rows, columns, seed).toFloatArray();
        return dtype == DType.FLOAT32 ? Tensor.from(values, rows, columns) : Tensor.from(values, dtype, rows, columns);
    }
}


