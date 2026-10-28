package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;
import java.util.Arrays;
import static org.junit.jupiter.api.Assertions.*;

class TensorShapeQueryTest {
    @Test
    void exactDimensionsAreIndependentOfDtypeAndRank() {
        for (DType dtype : DType.values()) {
            for (int rank = 0; rank <= 6; rank++) {
                int[] dimensions = new int[rank];
                Arrays.fill(dimensions, 2);
                Tensor tensor = new Tensor(dtype, dimensions);
                assertTrue(tensor.hasShape(dimensions));
                assertFalse(tensor.hasShape(Arrays.copyOf(dimensions, rank + 1)));
                if (rank > 0) assertFalse(tensor.hasShape(new int[rank]));
            }
        }
    }

    @Test
    void queryNeitherRetainsCallerDimensionsNorExposesInternalShape() {
        int[] dimensions = {2, 3};
        Tensor tensor = new Tensor(dimensions);
        assertTrue(tensor.hasShape(dimensions));
        dimensions[0] = 7;
        assertFalse(tensor.hasShape(dimensions));
        assertTrue(tensor.hasShape(2, 3));
        tensor.shape()[0] = 9;
        assertTrue(tensor.hasShape(2, 3));
    }

    @Test
    void logicalViewsAndEmptyAxesAreComparedWithoutFlattening() {
        Tensor view = Tensor.ones(DType.FLOAT32, 3, 2).slice(0, 1, 3).transpose(0, 1);
        assertTrue(view.hasShape(2, 2));
        assertFalse(view.hasShape(4));
        assertTrue(new Tensor(2, 0, 3).hasShape(2, 0, 3));
        assertFalse(new Tensor(2, 0, 3).hasShape(0));
        assertFalse(view.hasShape(-2, 2));
        assertFalse(view.hasShape((int[]) null));
    }

    @Test
    void deviceOnlyShapeQueryNeverAllocatesOrReadsHostValues() {
        Tensor tensor = Tensor.deviceOnly(DType.FLOAT16, 2, 3, 4);
        assertTrue(tensor.hasShape(2, 3, 4));
        assertFalse(tensor.hasShape(24));
        assertFalse(tensor.memory().hasHostStorage());
        assertNull(tensor.getGpuTag());
    }
}
