package io.github.kirstenali.deepj.tensor.cpu;

import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class CpuBackendReductionTest extends CpuBackendTestSupport {
    @Test
    void addBroadcastCols_shouldBroadcastDownColumns() {
        Tensor a = Tensor.from2D(new float[][]{
                {1, 2, 3},
                {4, 5, 6}
        });
        Tensor col = Tensor.from2D(new float[][]{
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
        Tensor a = Tensor.from2D(new float[][]{
                {11, 12, 13},
                {104, 105, 106}
        });
        Tensor col = Tensor.from2D(new float[][]{
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
        Tensor a = Tensor.from2D(new float[][]{
                {1, 2},
                {3, 4}
        });
        Tensor col = Tensor.from2D(new float[][]{
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
        Tensor a = Tensor.from2D(new float[][]{
                {10, 20},
                {300, 400}
        });
        Tensor col = Tensor.from2D(new float[][]{
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
        Tensor a = Tensor.from2D(new float[][]{
                {1, 2, 3},
                {4, 5, 6}
        });
        Tensor invalidCol = Tensor.from2D(new float[][]{
                {1, 2}
        });

        assertThrows(IllegalArgumentException.class, () -> backend.addBroadcastCols(a, invalidCol));
    }

    @Test
    void subtractBroadcastCols_shouldThrowOnInvalidShape() {
        Tensor a = Tensor.from2D(new float[][]{
                {1, 2, 3},
                {4, 5, 6}
        });
        Tensor invalidCol = Tensor.from2D(new float[][]{
                {1, 2}
        });

        assertThrows(IllegalArgumentException.class, () -> backend.subtractBroadcastCols(a, invalidCol));
    }

    @Test
    void multiplyBroadcastCols_shouldThrowOnInvalidShape() {
        Tensor a = Tensor.from2D(new float[][]{
                {1, 2, 3},
                {4, 5, 6}
        });
        Tensor invalidCol = Tensor.from2D(new float[][]{
                {1, 2}
        });

        assertThrows(IllegalArgumentException.class, () -> backend.multiplyBroadcastCols(a, invalidCol));
    }

    @Test
    void divideBroadcastCols_shouldThrowOnInvalidShape() {
        Tensor a = Tensor.from2D(new float[][]{
                {1, 2, 3},
                {4, 5, 6}
        });
        Tensor invalidCol = Tensor.from2D(new float[][]{
                {1, 2}
        });

        assertThrows(IllegalArgumentException.class, () -> backend.divideBroadcastCols(a, invalidCol));
    }

    @Test
    void sumRows_shouldReduceRowsIntoSingleRow() {
        Tensor a = Tensor.from2D(new float[][]{
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
        Tensor a = Tensor.from2D(new float[][]{
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
        Tensor a = Tensor.from2D(new float[][]{
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
        Tensor a = Tensor.from2D(new float[][]{
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
        Tensor a = Tensor.from2D(new float[][]{
                {1, 2, 3},
                {2, 2, 2}
        });

        Tensor result = backend.varianceAlongRows(a);

        assertTensorEquals(new double[][]{
                {2.0f / 3.0f},
                {0.0f}
        }, result);
    }

    @Test
    void transpose_shouldSwapRowsAndCols() {
        Tensor a = Tensor.from2D(new float[][]{
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
        Tensor a = Tensor.from2D(new float[][]{
                {-2, 0.5f, 3},
                {10, -1, 2}
        });

        Tensor result = backend.clamp(a, 0, 2);

        assertTensorEquals(new double[][]{
                {0, 0.5f, 2},
                {2, 0, 2}
        }, result);
    }

    @Test
    void sqrt_shouldApplyElementwise() {
        Tensor a = Tensor.from2D(new float[][]{
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
        Tensor a = Tensor.from2D(new float[][]{
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
        Tensor a = Tensor.from2D(new float[][]{
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
        Tensor a = Tensor.from2D(new float[][]{
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
        Tensor a = Tensor.from2D(new float[][]{
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
        Tensor a = Tensor.from2D(new float[][]{
                {1, 2, 3},
                {4, 5, 6}
        });

        assertEquals(21.0f, backend.sum(a), EPS);
    }

    @Test
    void sumAbs_shouldReturnAbsoluteTotalOfAllElements() {
        Tensor a = Tensor.from2D(new float[][]{
                {-1, -2, 3},
                {-4, 5, -6}
        });

        assertEquals(21.0f, backend.sumAbs(a), EPS);
    }
}
