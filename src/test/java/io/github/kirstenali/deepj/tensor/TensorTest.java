package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class TensorTest {

    @Test
    void constructorRejectsInvalidShapes() {
        assertThrows(IllegalArgumentException.class, () -> new Tensor(0, 2));
        assertThrows(IllegalArgumentException.class, () -> new Tensor(2, 0));
        assertThrows(IllegalArgumentException.class, () -> new Tensor(Integer.MAX_VALUE, 2));
    }

    @Test
    void from2DRejectsEmptyAndRaggedData() {
        assertThrows(IllegalArgumentException.class, () -> Tensor.from2D(new float[0][]));
        assertThrows(IllegalArgumentException.class,
                () -> Tensor.from2D(new float[][]{{1.0f}, {2.0f, 3.0f}}));
    }

    @Test
    void elementAccessRejectsColumnThatWouldAliasNextRow() {
        Tensor tensor = Tensor.from2D(new float[][]{{1, 2}, {3, 4}});

        assertThrows(IndexOutOfBoundsException.class, () -> tensor.get(0, 2));
        assertThrows(IndexOutOfBoundsException.class, () -> tensor.set(0, 2, 9));
        assertEquals(3.0f, tensor.get(1, 0));
    }

    @Test
    void rowAccessRejectsPastEnd() {
        Tensor tensor = Tensor.from2D(new float[][]{{1, 2}});

        assertThrows(IndexOutOfBoundsException.class, () -> tensor.rowData(1));
        assertThrows(IndexOutOfBoundsException.class, () -> tensor.getRow(1));
    }

    @Test
    void sliceRowsRejectsMismatchedWidth() {
        Tensor tensor = Tensor.from2D(new float[][]{{1, 2}, {3, 4}});

        assertThrows(IllegalArgumentException.class,
                () -> Tensor.sliceRows(tensor, new int[]{1}, 1));
    }
}
