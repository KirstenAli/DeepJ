package io.github.kirstenali.deepj.tensor.cpu;

import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class CpuBackendTest {

    private static final double EPS = 1e-9;

    private CpuBackend backend;

    @BeforeEach
    void setUp() {
        backend = new CpuBackend();
    }

    private void assertTensorEquals(double[][] expected, Tensor actual) {
        assertEquals(expected.length, actual.rows, "Row count mismatch");
        assertEquals(expected[0].length, actual.cols, "Column count mismatch");

        for (int r = 0; r < expected.length; r++) {
            for (int c = 0; c < expected[0].length; c++) {
                assertEquals(expected[r][c], actual.data[r][c], EPS,
                        "Mismatch at [" + r + "][" + c + "]");
            }
        }
    }

    @Test
    void zeros_shouldCreateAllZeros() {
        Tensor t = backend.zeros(2, 3);

        assertEquals(2, t.rows);
        assertEquals(3, t.cols);
        assertTensorEquals(new double[][]{
                {0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0}
        }, t);
    }

    @Test
    void ones_shouldCreateAllOnes() {
        Tensor t = backend.ones(2, 3);

        assertTensorEquals(new double[][]{
                {1.0, 1.0, 1.0},
                {1.0, 1.0, 1.0}
        }, t);
    }

    @Test
    void random_shouldBeDeterministicWithSeed() {
        Random r1 = new Random(123);
        Random r2 = new Random(123);

        Tensor t1 = backend.random(2, 3, r1);
        Tensor t2 = backend.random(2, 3, r2);

        for (int r = 0; r < t1.rows; r++) {
            for (int c = 0; c < t1.cols; c++) {
                assertEquals(t1.data[r][c], t2.data[r][c], EPS);
            }
        }
    }

    @Test
    void causalMask_shouldMaskUpperTriangle() {
        Tensor mask = backend.causalMask(4);

        assertTensorEquals(new double[][]{
                {0.0, -1e9, -1e9, -1e9},
                {0.0, 0.0, -1e9, -1e9},
                {0.0, 0.0, 0.0, -1e9},
                {0.0, 0.0, 0.0, 0.0}
        }, mask);
    }

    @Test
    void flattenAndUnflatten_shouldRoundTrip() {
        Tensor t = new Tensor(new double[][]{
                {1, 2, 3},
                {4, 5, 6}
        });

        double[] flat = backend.flattenTensor(t);
        assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6}, flat, EPS);

        Tensor rebuilt = backend.unflattenToTensor(flat, 2, 3);
        assertTensorEquals(new double[][]{
                {1, 2, 3},
                {4, 5, 6}
        }, rebuilt);
    }

    @Test
    void matmul_shouldMultiplyMatrices() {
        Tensor a = new Tensor(new double[][]{
                {1, 2, 3},
                {4, 5, 6}
        });
        Tensor b = new Tensor(new double[][]{
                {7, 8},
                {9, 10},
                {11, 12}
        });

        Tensor result = backend.matmul(a, b);

        assertTensorEquals(new double[][]{
                {58, 64},
                {139, 154}
        }, result);
    }

    @Test
    void matmul_shouldThrowOnShapeMismatch() {
        Tensor a = new Tensor(new double[][]{
                {1, 2},
                {3, 4}
        });
        Tensor b = new Tensor(new double[][]{
                {1, 2, 3}
        });

        assertThrows(IllegalArgumentException.class, () -> backend.matmul(a, b));
    }

    @Test
    void add_shouldAddElementwise() {
        Tensor a = new Tensor(new double[][]{
                {1, 2},
                {3, 4}
        });
        Tensor b = new Tensor(new double[][]{
                {10, 20},
                {30, 40}
        });

        Tensor result = backend.add(a, b);

        assertTensorEquals(new double[][]{
                {11, 22},
                {33, 44}
        }, result);
    }

    @Test
    void subtract_shouldSubtractElementwise() {
        Tensor a = new Tensor(new double[][]{
                {10, 20},
                {30, 40}
        });
        Tensor b = new Tensor(new double[][]{
                {1, 2},
                {3, 4}
        });

        Tensor result = backend.subtract(a, b);

        assertTensorEquals(new double[][]{
                {9, 18},
                {27, 36}
        }, result);
    }

    @Test
    void multiply_shouldMultiplyElementwise() {
        Tensor a = new Tensor(new double[][]{
                {1, 2},
                {3, 4}
        });
        Tensor b = new Tensor(new double[][]{
                {10, 20},
                {30, 40}
        });

        Tensor result = backend.multiply(a, b);

        assertTensorEquals(new double[][]{
                {10, 40},
                {90, 160}
        }, result);
    }

    @Test
    void divide_shouldDivideElementwise() {
        Tensor a = new Tensor(new double[][]{
                {10, 20},
                {30, 40}
        });
        Tensor b = new Tensor(new double[][]{
                {2, 4},
                {5, 8}
        });

        Tensor result = backend.divide(a, b);

        assertTensorEquals(new double[][]{
                {5, 5},
                {6, 5}
        }, result);
    }

    @Test
    void add_shouldThrowOnShapeMismatch() {
        Tensor a = new Tensor(new double[][]{
                {1, 2}
        });
        Tensor b = new Tensor(new double[][]{
                {1},
                {2}
        });

        assertThrows(IllegalArgumentException.class, () -> backend.add(a, b));
    }

    @Test
    void subtract_shouldThrowOnShapeMismatch() {
        Tensor a = new Tensor(new double[][]{
                {1, 2}
        });
        Tensor b = new Tensor(new double[][]{
                {1},
                {2}
        });

        assertThrows(IllegalArgumentException.class, () -> backend.subtract(a, b));
    }

    @Test
    void multiply_shouldThrowOnShapeMismatch() {
        Tensor a = new Tensor(new double[][]{
                {1, 2}
        });
        Tensor b = new Tensor(new double[][]{
                {1},
                {2}
        });

        assertThrows(IllegalArgumentException.class, () -> backend.multiply(a, b));
    }

    @Test
    void divide_shouldThrowOnShapeMismatch() {
        Tensor a = new Tensor(new double[][]{
                {1, 2}
        });
        Tensor b = new Tensor(new double[][]{
                {1},
                {2}
        });

        assertThrows(IllegalArgumentException.class, () -> backend.divide(a, b));
    }

    @Test
    void addRowVector_shouldBroadcastAcrossRows() {
        Tensor a = new Tensor(new double[][]{
                {1, 2, 3},
                {4, 5, 6}
        });
        Tensor row = new Tensor(new double[][]{
                {10, 20, 30}
        });

        Tensor result = backend.addRowVector(a, row);

        assertTensorEquals(new double[][]{
                {11, 22, 33},
                {14, 25, 36}
        }, result);
    }

    @Test
    void addBroadcastRows_shouldBroadcastAcrossRows() {
        Tensor a = new Tensor(new double[][]{
                {1, 2, 3},
                {4, 5, 6}
        });
        Tensor row = new Tensor(new double[][]{
                {10, 20, 30}
        });

        Tensor result = backend.addBroadcastRows(a, row);

        assertTensorEquals(new double[][]{
                {11, 22, 33},
                {14, 25, 36}
        }, result);
    }

    @Test
    void multiplyBroadcastRows_shouldBroadcastAcrossRows() {
        Tensor a = new Tensor(new double[][]{
                {1, 2, 3},
                {4, 5, 6}
        });
        Tensor row = new Tensor(new double[][]{
                {10, 20, 30}
        });

        Tensor result = backend.multiplyBroadcastRows(a, row);

        assertTensorEquals(new double[][]{
                {10, 40, 90},
                {40, 100, 180}
        }, result);
    }

    @Test
    void addRowVector_shouldThrowOnInvalidShape() {
        Tensor a = new Tensor(new double[][]{
                {1, 2, 3},
                {4, 5, 6}
        });
        Tensor invalidRow = new Tensor(new double[][]{
                {1, 2},
                {3, 4}
        });

        assertThrows(IllegalArgumentException.class, () -> backend.addRowVector(a, invalidRow));
    }

    @Test
    void multiplyBroadcastRows_shouldThrowOnInvalidShape() {
        Tensor a = new Tensor(new double[][]{
                {1, 2, 3},
                {4, 5, 6}
        });
        Tensor invalidRow = new Tensor(new double[][]{
                {1, 2},
                {3, 4}
        });

        assertThrows(IllegalArgumentException.class, () -> backend.multiplyBroadcastRows(a, invalidRow));
    }

    @Test
    void addBroadcastCols_shouldBroadcastDownColumns() {
        Tensor a = new Tensor(new double[][]{
                {1, 2, 3},
                {4, 5, 6}
        });
        Tensor col = new Tensor(new double[][]{
                {10},
                {100}
        });

        Tensor result = backend.addBroadcastCols(a, col);

        assertTensorEquals(new double[][]{
                {11, 12, 13},
                {104, 105, 106}
        }, result);
    }

    @Test
    void subtractBroadcastCols_shouldBroadcastDownColumns() {
        Tensor a = new Tensor(new double[][]{
                {11, 12, 13},
                {104, 105, 106}
        });
        Tensor col = new Tensor(new double[][]{
                {10},
                {100}
        });

        Tensor result = backend.subtractBroadcastCols(a, col);

        assertTensorEquals(new double[][]{
                {1, 2, 3},
                {4, 5, 6}
        }, result);
    }

    @Test
    void multiplyBroadcastCols_shouldBroadcastDownColumns() {
        Tensor a = new Tensor(new double[][]{
                {1, 2},
                {3, 4}
        });
        Tensor col = new Tensor(new double[][]{
                {10},
                {100}
        });

        Tensor result = backend.multiplyBroadcastCols(a, col);

        assertTensorEquals(new double[][]{
                {10, 20},
                {300, 400}
        }, result);
    }

    @Test
    void divideBroadcastCols_shouldBroadcastDownColumns() {
        Tensor a = new Tensor(new double[][]{
                {10, 20},
                {300, 400}
        });
        Tensor col = new Tensor(new double[][]{
                {10},
                {100}
        });

        Tensor result = backend.divideBroadcastCols(a, col);

        assertTensorEquals(new double[][]{
                {1, 2},
                {3, 4}
        }, result);
    }

    @Test
    void addBroadcastCols_shouldThrowOnInvalidShape() {
        Tensor a = new Tensor(new double[][]{
                {1, 2, 3},
                {4, 5, 6}
        });
        Tensor invalidCol = new Tensor(new double[][]{
                {1, 2}
        });

        assertThrows(IllegalArgumentException.class, () -> backend.addBroadcastCols(a, invalidCol));
    }

    @Test
    void subtractBroadcastCols_shouldThrowOnInvalidShape() {
        Tensor a = new Tensor(new double[][]{
                {1, 2, 3},
                {4, 5, 6}
        });
        Tensor invalidCol = new Tensor(new double[][]{
                {1, 2}
        });

        assertThrows(IllegalArgumentException.class, () -> backend.subtractBroadcastCols(a, invalidCol));
    }

    @Test
    void multiplyBroadcastCols_shouldThrowOnInvalidShape() {
        Tensor a = new Tensor(new double[][]{
                {1, 2, 3},
                {4, 5, 6}
        });
        Tensor invalidCol = new Tensor(new double[][]{
                {1, 2}
        });

        assertThrows(IllegalArgumentException.class, () -> backend.multiplyBroadcastCols(a, invalidCol));
    }

    @Test
    void divideBroadcastCols_shouldThrowOnInvalidShape() {
        Tensor a = new Tensor(new double[][]{
                {1, 2, 3},
                {4, 5, 6}
        });
        Tensor invalidCol = new Tensor(new double[][]{
                {1, 2}
        });

        assertThrows(IllegalArgumentException.class, () -> backend.divideBroadcastCols(a, invalidCol));
    }

    @Test
    void sumRows_shouldReduceRowsIntoSingleRow() {
        Tensor a = new Tensor(new double[][]{
                {1, 2, 3},
                {4, 5, 6}
        });

        Tensor result = backend.sumRows(a);

        assertTensorEquals(new double[][]{
                {5, 7, 9}
        }, result);
    }

    @Test
    void sumAlongRows_shouldSumEachRow() {
        Tensor a = new Tensor(new double[][]{
                {1, 2, 3},
                {4, 5, 6}
        });

        Tensor result = backend.sumAlongRows(a);

        assertTensorEquals(new double[][]{
                {6},
                {15}
        }, result);
    }

    @Test
    void sumAlongCols_shouldAliasSumRows() {
        Tensor a = new Tensor(new double[][]{
                {1, 2},
                {3, 4}
        });

        Tensor result = backend.sumAlongCols(a);

        assertTensorEquals(new double[][]{
                {4, 6}
        }, result);
    }

    @Test
    void meanAlongRows_shouldComputeRowMeans() {
        Tensor a = new Tensor(new double[][]{
                {2, 4, 6},
                {1, 3, 5}
        });

        Tensor result = backend.meanAlongRows(a);

        assertTensorEquals(new double[][]{
                {4},
                {3}
        }, result);
    }

    @Test
    void varianceAlongRows_shouldComputeRowVariance() {
        Tensor a = new Tensor(new double[][]{
                {1, 2, 3},
                {2, 2, 2}
        });

        Tensor result = backend.varianceAlongRows(a);

        assertTensorEquals(new double[][]{
                {2.0 / 3.0},
                {0.0}
        }, result);
    }

    @Test
    void transpose_shouldSwapRowsAndCols() {
        Tensor a = new Tensor(new double[][]{
                {1, 2, 3},
                {4, 5, 6}
        });

        Tensor result = backend.transpose(a);

        assertTensorEquals(new double[][]{
                {1, 4},
                {2, 5},
                {3, 6}
        }, result);
    }

    @Test
    void clamp_shouldLimitValuesToRange() {
        Tensor a = new Tensor(new double[][]{
                {-2, 0.5, 3},
                {10, -1, 2}
        });

        Tensor result = backend.clamp(a, 0, 2);

        assertTensorEquals(new double[][]{
                {0, 0.5, 2},
                {2, 0, 2}
        }, result);
    }

    @Test
    void sqrt_shouldApplyElementwise() {
        Tensor a = new Tensor(new double[][]{
                {1, 4},
                {9, 16}
        });

        Tensor result = backend.sqrt(a);

        assertTensorEquals(new double[][]{
                {1, 2},
                {3, 4}
        }, result);
    }

    @Test
    void pow_shouldApplyElementwise() {
        Tensor a = new Tensor(new double[][]{
                {1, 2},
                {3, 4}
        });

        Tensor result = backend.pow(a, 2);

        assertTensorEquals(new double[][]{
                {1, 4},
                {9, 16}
        }, result);
    }

    @Test
    void multiplyScalar_shouldApplyToAllElements() {
        Tensor a = new Tensor(new double[][]{
                {1, 2},
                {3, 4}
        });

        Tensor result = backend.multiplyScalar(a, 10);

        assertTensorEquals(new double[][]{
                {10, 20},
                {30, 40}
        }, result);
    }

    @Test
    void addScalar_shouldApplyToAllElements() {
        Tensor a = new Tensor(new double[][]{
                {1, 2},
                {3, 4}
        });

        Tensor result = backend.addScalar(a, 5);

        assertTensorEquals(new double[][]{
                {6, 7},
                {8, 9}
        }, result);
    }

    @Test
    void divideScalar_shouldApplyToAllElements() {
        Tensor a = new Tensor(new double[][]{
                {10, 20},
                {30, 40}
        });

        Tensor result = backend.divideScalar(a, 10);

        assertTensorEquals(new double[][]{
                {1, 2},
                {3, 4}
        }, result);
    }

    @Test
    void sum_shouldReturnTotalOfAllElements() {
        Tensor a = new Tensor(new double[][]{
                {1, 2, 3},
                {4, 5, 6}
        });

        assertEquals(21.0, backend.sum(a), EPS);
    }

    @Test
    void sumAbs_shouldReturnAbsoluteTotalOfAllElements() {
        Tensor a = new Tensor(new double[][]{
                {-1, -2, 3},
                {-4, 5, -6}
        });

        assertEquals(21.0, backend.sumAbs(a), EPS);
    }
}