package io.github.kirstenali.deepj.tensor.cpu;

import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

abstract class CpuBackendTestSupport {

    protected static final double EPS = 1e-6f;

    protected CpuBackend backend;

    @BeforeEach
    void setUp() {
        backend = new CpuBackend();
    }

    protected void assertTensorEquals(double[][] expected, Tensor actual) {
        assertEquals(expected.length, actual.rows, "Row count mismatch");
        assertEquals(expected[0].length, actual.cols, "Column count mismatch");

        for (int r = 0; r < expected.length; r++) {
            for (int c = 0; c < expected[0].length; c++) {
                assertEquals(expected[r][c], actual.data[r * actual.cols + c], EPS,
                        "Mismatch at [" + r + "][" + c + "]");
            }
        }
    }


}
