package io.github.kirstenali.deepj.tensor.cpu;

import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class CpuBackendTest extends CpuBackendTestSupport {

    @Test
    void zeros_shouldCreateAllZeros() {
        Tensor t = backend.zeros(2, 3);

        assertEquals(2, t.matrixRows());
        assertEquals(3, t.matrixCols());
        assertTensorEquals(new double[][]{
                {0.0f, 0.0f, 0.0f},
                {0.0f, 0.0f, 0.0f}
        }, t);
    }

    @Test
    void ones_shouldCreateAllOnes() {
        Tensor t = backend.ones(2, 3);

        assertTensorEquals(new double[][]{
                {1.0f, 1.0f, 1.0f},
                {1.0f, 1.0f, 1.0f}
        }, t);
    }

    @Test
    void random_shouldBeDeterministicWithSeed() {
        Random r1 = new Random(123);
        Random r2 = new Random(123);

        Tensor t1 = backend.random(2, 3, r1);
        Tensor t2 = backend.random(2, 3, r2);

        for (int r = 0; r < t1.matrixRows(); r++) {
            for (int c = 0; c < t1.matrixCols(); c++) {
                assertEquals(t1.rawData()[r * t1.matrixCols() + c], t2.rawData()[r * t2.matrixCols() + c], EPS);
            }
        }
    }

    @Test
    void causalMask_shouldMaskUpperTriangle() {
        Tensor mask = backend.causalMask(4);

        assertTensorEquals(new double[][]{
                {0.0f, -1e9f, -1e9f, -1e9f},
                {0.0f, 0.0f, -1e9f, -1e9f},
                {0.0f, 0.0f, 0.0f, -1e9f},
                {0.0f, 0.0f, 0.0f, 0.0f}
        }, mask);
    }

    @Test
    void matmul_shouldMultiplyMatrices() {
        Tensor a = Tensor.from2D(new float[][]{
                {1, 2, 3},
                {4, 5, 6}
        });
        Tensor b = Tensor.from2D(new float[][]{
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
        Tensor a = Tensor.from2D(new float[][]{
                {1, 2},
                {3, 4}
        });
        Tensor b = Tensor.from2D(new float[][]{
                {1, 2, 3}
        });

        assertThrows(IllegalArgumentException.class, () -> backend.matmul(a, b));
    }

    @Test
    void add_shouldAddElementwise() {
        Tensor a = Tensor.from2D(new float[][]{
                {1, 2},
                {3, 4}
        });
        Tensor b = Tensor.from2D(new float[][]{
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
        Tensor a = Tensor.from2D(new float[][]{
                {10, 20},
                {30, 40}
        });
        Tensor b = Tensor.from2D(new float[][]{
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
        Tensor a = Tensor.from2D(new float[][]{
                {1, 2},
                {3, 4}
        });
        Tensor b = Tensor.from2D(new float[][]{
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
        Tensor a = Tensor.from2D(new float[][]{
                {10, 20},
                {30, 40}
        });
        Tensor b = Tensor.from2D(new float[][]{
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
    void add_shouldRejectIncompatibleBroadcastShapes() {
        assertIncompatibleBroadcast(backend::add);
    }

    @Test
    void subtract_shouldRejectIncompatibleBroadcastShapes() {
        assertIncompatibleBroadcast(backend::subtract);
    }

    @Test
    void multiply_shouldRejectIncompatibleBroadcastShapes() {
        assertIncompatibleBroadcast(backend::multiply);
    }

    @Test
    void divide_shouldRejectIncompatibleBroadcastShapes() {
        assertIncompatibleBroadcast(backend::divide);
    }

    private static void assertIncompatibleBroadcast(java.util.function.BiFunction<Tensor, Tensor, Tensor> operation) {
        Tensor left = Tensor.from2D(new float[][]{{1, 2}});
        Tensor right = Tensor.from2D(new float[][]{{1, 2, 3}, {4, 5, 6}});
        assertThrows(IllegalArgumentException.class, () -> operation.apply(left, right));
    }

    @Test
    void rowAndColumnVectorsUseTrailingBroadcasts() {
        Tensor left = Tensor.from2D(new float[][]{{1, 2}});
        Tensor right = Tensor.from2D(new float[][]{{1}, {2}});
        assertTensorEquals(new double[][]{{2, 3}, {3, 4}}, backend.add(left, right));
        assertTensorEquals(new double[][]{{0, 1}, {-1, 0}}, backend.subtract(left, right));
        assertTensorEquals(new double[][]{{1, 2}, {2, 4}}, backend.multiply(left, right));
        assertTensorEquals(new double[][]{{1, 2}, {0.5, 1}}, backend.divide(left, right));
    }

    @Test
    void addRowVector_shouldBroadcastAcrossRows() {
        Tensor a = Tensor.from2D(new float[][]{
                {1, 2, 3},
                {4, 5, 6}
        });
        Tensor row = Tensor.from2D(new float[][]{
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
        Tensor a = Tensor.from2D(new float[][]{
                {1, 2, 3},
                {4, 5, 6}
        });
        Tensor row = Tensor.from2D(new float[][]{
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
        Tensor a = Tensor.from2D(new float[][]{
                {1, 2, 3},
                {4, 5, 6}
        });
        Tensor row = Tensor.from2D(new float[][]{
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
        Tensor a = Tensor.from2D(new float[][]{
                {1, 2, 3},
                {4, 5, 6}
        });
        Tensor invalidRow = Tensor.from2D(new float[][]{
                {1, 2},
                {3, 4}
        });

        assertThrows(IllegalArgumentException.class, () -> backend.addRowVector(a, invalidRow));
    }

    @Test
    void multiplyBroadcastRows_shouldThrowOnInvalidShape() {
        Tensor a = Tensor.from2D(new float[][]{
                {1, 2, 3},
                {4, 5, 6}
        });
        Tensor invalidRow = Tensor.from2D(new float[][]{
                {1, 2},
                {3, 4}
        });

        assertThrows(IllegalArgumentException.class, () -> backend.multiplyBroadcastRows(a, invalidRow));
    }


}
